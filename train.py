import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms

from torchvision.ops import sigmoid_focal_loss

from pathlib import Path
from tqdm import tqdm
# from monai.losses import DiceCELoss

from finetune_utils.load_config import get_config
from finetune_utils.load_logger import Logger
from finetune_utils.load_checkpoint import get_sam_vit_t
from finetune_utils.datasets import SAMDataset
from finetune_utils.loss import DiceLoss, batch_iou
from finetune_utils.visualization import overlay_mask_on_image
from finetune_utils.save_checkpoint import save_checkpoint
from finetune_utils.schedular import LinearWarmup
import numpy as np

torch.backends.cudnn.benchmark = True

def main(args):
    # Assert that a CUDA device is available
    assert torch.cuda.is_available(), "CUDA is not available."

    # Create dataset and dataloader for training and validation
    train_dataset = SAMDataset(root_dir=args.dataset.train_dataset, transform=[transform_img, transform_mask], max_bbox_shift=args.dataset.max_bbox_shift)
    val_dataset = SAMDataset(root_dir=args.dataset.val_dataset, transform=[transform_img, transform_mask], max_bbox_shift=args.dataset.max_bbox_shift)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training or validation dataset is empty. Please check the dataset paths and contents.")
    train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size, num_workers=args.dataset.num_workers, shuffle=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.train.batch_size, num_workers=args.dataset.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    

    test_dataset = SAMDataset(root_dir=args.dataset.test_dataset, transform=[transform_img, transform_mask], max_bbox_shift=args.dataset.max_bbox_shift)
    test_loader = DataLoader(test_dataset, batch_size=args.train.batch_size, num_workers=args.dataset.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    # Define checkpoint and saving paths
    checkpoint_path = Path(args.model.checkpoint_path)
    save_path = Path(args.model.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize the logger
    logger = Logger(save_path / 'training.log').get_logger()

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Load the MobileSAM checkpoint and move it to CUDA
    # TO DO: resume checkpoint from last.pth
    model = get_sam_vit_t(checkpoint=checkpoint_path, resume=args.train.resume).cuda()
    
    # Conditionally freeze layers based on args
    for param in model.image_encoder.parameters():
        param.requires_grad = not args.freeze.freeze_image_encoder
    for param in model.prompt_encoder.parameters():
        param.requires_grad = not args.freeze.freeze_prompt_encoder
    for param in model.mask_decoder.parameters():
        param.requires_grad = not args.freeze.freeze_mask_decoder

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.train.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train.epochs * len(train_dataset))
    warmup_scheduler = LinearWarmup(optimizer, warmup_period=args.train.warmup_step)
    

    criterion_MSE = nn.MSELoss()
    criterion_Dice = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')

    # Initialize TensorBoard writer for logging
    writer = SummaryWriter()

    # Initialize the best validation loss variable
    best_val_loss = float('inf')

    # Main training loop
    if args.train.status:
        logger.info("Starting training...")
        for epoch in range(args.train.epochs):
            # Train for one epoch
            train_loss = train_epoch(train_loader, model, optimizer, criterion_MSE, criterion_Dice, epoch, writer, scaler, lr_scheduler, warmup_scheduler)
            logger.info(f"Epoch {epoch+1}/{args.train.epochs}, Train Loss: {train_loss:.4f}")

            # Validate and save the model at specified intervals
            if (epoch + 1) % args.train.val_freq == 0:
                val_loss = val_epoch(val_loader, model, criterion_MSE, criterion_Dice, epoch, writer, scaler)
                logger.info(f"Epoch {epoch+1}/{args.train.epochs}, Val Loss: {val_loss:.4f}")

                # Save the best model based on validation loss
                # the best model could be used like the original MobileSAM checkpoint without any modification
                is_best = val_loss < best_val_loss
                save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, is_best, save_path)
                if is_best:
                    best_val_loss = val_loss
            
        
        ## test the model after training
        logger.info("Training completed. Loading the best model for testing...")
        
        test_iou, test_dice = test_epoch(test_loader, model, threshold=args.visual.IOU_threshold)
        logger.info(f"Test IoU: {test_iou:.4f}, Test Dice: {test_dice:.4f}")
    else:

        logger.info("Training is disabled. Skipping training loop.")
        logger.info("Loading the best model for testing...")
        # # Load the best model for testing
        # best_model_path = save_path / 'best.pth'
        # if best_model_path.exists():
        #     model = get_sam_vit_t(checkpoint=best_model_path, resume=args.train.resume).cuda()
        #     logger.info(f"Loaded best model from {best_model_path}")
        # else:
        #     logger.warning(f"No best model found at {best_model_path}. Testing with the current model state.")
        model.eval()
        logger.info("Starting testing...")
        test_iou, test_dice = test_epoch(test_loader, model, threshold=args.visual.IOU_threshold)
        logger.info(f"Test IoU: {test_iou:.4f}, Test Dice: {test_dice:.4f}")

def train_epoch(dataloader, model, optimizer, criterion_MSE, criterion_Dice, epoch, writer, scaler, lr_scheduler, warmup_scheduler):
    """Main training function."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Training", total=num_batches)

    for batch_idx, (image, mask, bbox) in enumerate(progress_bar):
        # Move input and target data to the GPU
        image, mask, bbox = image.cuda(non_blocking=True), mask.cuda(non_blocking=True), bbox.cuda(non_blocking=True)

        # Forward pass with mixed precision
        with autocast(enabled=args.train.bf16, dtype=torch.bfloat16):
            pred_mask, pred_IOU = model(image, bbox)
            iou = batch_iou(mask, torch.sigmoid(pred_mask))

            loss_focal = sigmoid_focal_loss(pred_mask, mask, reduction='mean')
            loss_dice = criterion_Dice(pred_mask, mask)
            loss_mse = criterion_MSE(pred_IOU, iou)
            loss = loss_focal * 20 + loss_dice + loss_mse

        # Backward pass and update model parameters
        scaler.scale(loss).backward()
        if batch_idx % args.train.gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            with warmup_scheduler.dampening():
                lr_scheduler.step()
        

        # Accumulate the loss for logging
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Calculate average training loss for the epoch
    average_loss = total_loss / num_batches
            
    # Log the training loss to TensorBoard
    for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_rate/group_{i}', param_group['lr'], epoch)
    writer.add_scalar('Training loss', average_loss, epoch)
    
    return average_loss

def val_epoch(dataloader, model, criterion_MSE, criterion_Dice, epoch, writer, scaler):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Validating", total=num_batches)

    # Evaluation mode: no gradients needed
    with torch.no_grad():
        for batch_idx, (image, mask, bbox) in enumerate(progress_bar):
            # Move input and target data to the GPU
            image, mask, bbox = image.cuda(), mask.cuda(), bbox.cuda()
            
            # Forward pass
            pred_mask, pred_IOU = model(image, bbox)
            iou = batch_iou(mask, torch.sigmoid(pred_mask))

            loss_focal = sigmoid_focal_loss(pred_mask, mask, reduction='mean')
            loss_dice = criterion_Dice(pred_mask, mask)
            loss_mse = criterion_MSE(pred_IOU, iou)
            loss = loss_focal * 20 + loss_dice + loss_mse

            # Accumulate the loss for logging
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            if args.visual.status:
                vis_image = image[0]
                vis_mask = pred_mask[0]
                vis_bbox = bbox[0]
                vis_mask = torch.sigmoid(vis_mask)
                mean = torch.tensor(MEAN, device=vis_image.device)
                std = torch.tensor(STD, device=vis_image.device)
                vis_image = vis_image * std[:, None, None] + mean[:, None, None]
                overlay_mask_on_image(vis_image, vis_mask, vis_bbox, threshold=args.visual.IOU_threshold, save_dir=args.visual.save_path, info=(epoch, batch_idx))

        # Calculate average validation loss for the epoch
        average_loss = total_loss / num_batches
            
     # Log the validation loss to TensorBoard
    writer.add_scalar('Val loss', average_loss, epoch)

    return average_loss

def test_epoch(dataloader, model, threshold=0.5):
    model.eval()
    ious, dices = [], []

    with torch.no_grad():
        for image, mask, bbox in tqdm(dataloader, desc="Testing"):
            image, mask, bbox = image.cuda(), mask.cuda(), bbox.cuda()
            pred_mask, _ = model(image, bbox)
            pred_mask = torch.sigmoid(pred_mask) > threshold

            intersection = (pred_mask & mask.bool()).float().sum((1, 2, 3))
            union = (pred_mask | mask.bool()).float().sum((1, 2, 3))
            dice = (2 * intersection) / (pred_mask.float().sum((1, 2, 3)) + mask.float().sum((1, 2, 3)) + 1e-6)
            iou = intersection / (union + 1e-6)

            ious.extend(iou.cpu().numpy())
            dices.extend(dice.cpu().numpy())

    print(f"Test IoU: {np.mean(ious):.4f}, Dice: {np.mean(dices):.4f}")
    return np.mean(ious), np.mean(dices)


if __name__ == '__main__':
    # Load configuration settings
    args = get_config()

    # Define the desired image size
    IMAGE_SIZE = (args.model.image_size, args.model.image_size)

    # Define image and mask transformations
    # The normalization is the same in mobile_sam\modeling\sam.py
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transform_img = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
    ])
    transform_mask = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    main(args)