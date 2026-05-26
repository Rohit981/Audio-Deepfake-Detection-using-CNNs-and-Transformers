from datasets import load_dataset
import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
from huggingface_hub import snapshot_download
from Resnet50 import CustomResnet50
from Trainer import Trainer
import torch.nn as nn  
from tqdm import tqdm,trange
from torchmetrics.classification import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import time
from Utils import utils
from CNNTrasnformer.CnnTrasnformer import CNNTrasnformer


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # snapshot_download(
    #     repo_id="RohitGENAICODER/ASVspoofLADataset",
    #     repo_type="dataset",
    #     local_dir="Datasets/LA",
    #     token="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"
    # )

    Data_root = r"D:\Deep Neural Network\ML-Audio_DeepFake\Datasets\LA"

    #Instantiate Datasets
    train_dataset = data.ASVpoofDataset(Data_root,part="train", precompute=False)
    val_dataset = data.ASVpoofDataset(Data_root, part="dev", precompute=False)
    test_dataset = data.ASVpoofDataset(Data_root, part="eval",precompute=False)

    # #Create a sampler for data balance
    sampler = train_dataset.data_balancing(train_dataset)
    
    #Initialize Batch Size
    batch_size = 64
    
    #Define Loss Fn and assign weights
    # train_labels = train_dataset.file_labels
    # count_0 = train_labels.count(0)
    # count_1 = train_labels.count(1)

    # pos_weight_value = torch.tensor([count_0 / count_1], dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    #Resnet Model
    Resnet_50 = CustomResnet50.Resnet50(1)
    Resnet_18 = CustomResnet50.Resnet18(1)

    #CNN Transformer Model
    CNN_Transformer = CNNTrasnformer()

    save_dir = r"D:\Deep Neural Network\ML-Audio_DeepFake\Models\Resnet"
    model_name = "Best Resnet Model"
    start_from_checkpoint = False

    trainer = Trainer.ModelTrainer(Resnet_50, device,loss_fn,learning_rate=1e-4, 
                                   batch_size=batch_size,sampler=sampler,
                                   save_dir=save_dir,model_name=model_name,
                                   start_from_checkpoint=start_from_checkpoint)

    #Set Data Loader
    trainer.set_data(train_set=train_dataset,test_set=test_dataset,val_set=val_dataset)

    #Train and val
    n_epochs = 50
    pbar = trange(trainer.start_epoch, n_epochs, leave=False, desc="Epoch")

    #Track of time for evaluation and learning rate history
    start_time = time.time()
    
    for epoch in pbar:
        trainer.Training()
        
        #Calculate evaluation
        _, train_acc,_,_, _, _, _= trainer.eval_model(train_test_val="train")
        val_loss, val_acc,val_f1,val_recall, _, _, _ = trainer.eval_model(train_test_val="val")

        train_loss = trainer.train_loss[-1]

        print(f" Train Accuracy: {train_acc:.4f} |"
              f" Train Loss: {train_loss:.4f} |"
              f" Val Accuracy: {val_acc:.4f} |"
              f" Val Loss: {val_loss:.4f} |"
              f"F1 score: {val_f1:.4f} |"
              f"Recall score: {val_recall:.4f}")
        
        #Check if the current validation accuracy is greater than the previous best
        if val_acc > trainer.best_valid_acc:
            trainer.save_checkpoint(epoch,val_acc)
    end_time = time.time()

    print(f"The highest validation accuracy was: {trainer.best_valid_acc:.4f}" )
    print("Training time %.2f seconds" %(end_time - start_time))

    #Call the evaluate function and pass the evaluation/test dataloader
    test_loss, test_acc, test_f1, test_recall, test_preds, test_labels, test_probs = trainer.eval_model(train_test_val="test")
    print(f"Test Accuracy: {test_acc:.4f}")

    #Plot Graph to visualize Train loss, acc and val loss, acc
    utils.Visualize_loss_acc(trainer.train_loss,
                             trainer.train_acc,
                             trainer.val_loss,
                             trainer.val_acc,
                             trainer.lr_history)
        
    #Calculate Confusion Matrix
    confusion_matrix = ConfusionMatrix(task='binary')
    conf_mat = confusion_matrix(torch.tensor(test_labels), 
                                torch.tensor(test_preds))
    #Plot Confussion Matrix
    fig, ax = plot_confusion_matrix(conf_mat.numpy(),
                                    class_names=['bonafide', 'spoof'],
                                    figsize=(12,9))
    plt.show()
    

    #Plot ROC curve,AUC score and classification Report
    utils.ROC_AUC_Values(test_labels,test_probs)

    

    
    #Visualize Augmentation
    # original_img,_ = dev_dataset[0]
    # augmented_img,_ = train_dataset[0]

    # plt.figure(figsize=(12,5))

    # plt.subplot(1,3,1)
    # plt.imshow(
    #     original_img.squeeze().numpy(),
    #     aspect='auto',
    #     origin='lower',
    #     cmap='magma'
    #     )
    # plt.title('Original Image')

    # plt.subplot(1,3,2)
    # plt.imshow(
    #     augmented_img.squeeze().numpy(),
    #     aspect='auto',
    #     origin='lower',
    #     cmap='magma'
    #     )
    # plt.title('Augmented Image')

    # plt.subplot(1,3,3)
    # difference = augmented_img - original_img
    # plt.imshow(
    #     difference.squeeze().numpy(),
    #     aspect='auto',
    #     origin='lower',
    #     cmap='magma'
    # )
    # plt.title("Injected Noise")

    # plt.tight_layout()
    # plt.show()
   



if __name__ == "__main__":
    main()