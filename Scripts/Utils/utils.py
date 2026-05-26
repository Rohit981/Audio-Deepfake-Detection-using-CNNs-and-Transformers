import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAUROC,BinaryROC
from sklearn.metrics import roc_curve,classification_report,roc_auc_score
import numpy as np


#Visualize Train and Test loss and Accuracy
def Visualize_loss_acc(train_loss, train_acc, val_loss, val_acc, lr_history=[]):
    plt.figure(figsize=(12,5))
    epochs = range(1, len(train_loss) + 1)

    #Train Loss
    plt.subplot(2,3,1)
    plt.plot(epochs,train_loss,label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()

    #Train Accuracy
    plt.subplot(2,3,2)
    plt.plot(epochs,train_acc,label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train Accuracy")
    plt.legend()

    #Test Loss
    plt.subplot(2,3,3)
    plt.plot(epochs,val_loss,label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Val Loss")
    plt.legend()

    #Test Accuracy
    plt.subplot(2,3,4)
    plt.plot(epochs,val_acc,label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Val Accuracy")
    plt.legend()

    #Plot Learning Rate vs Epoch
    steps = range(len(lr_history))
    plt.subplot(2,3,5)
    plt.plot(steps,lr_history, label="LR")
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.legend()

    plt.tight_layout()
    plt.show()

def ROC_AUC_Values(y_true,y_probs):
    #Calculate the ROC curve coordinates
    fpr, tpr, thresholds = roc_curve(y_true,y_probs,pos_label=1)
    fnr = 1 - tpr

    #Calculate ROC AUC Score
    auc_score = roc_auc_score(y_true,y_probs)
    print(f"ROC AUC Score:{auc_score:.4f}")

    #Locate the intersection point where FPR ~= FNR
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[idx]
    optimal_threshold = thresholds[idx]

    print("\n" + "="*40)
    print(f" Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f" Optimal Decision Threshold: {optimal_threshold:.6f}")
    print("="*40 + "\n")
    
    #Apply the optimal threshold to map new predictions
    tuned_preds = [1 if p > optimal_threshold else 0 for p in y_probs]
    
    #Generate the adjusted report
    print("Tuned Classification Report:")
    print(classification_report(y_true, tuned_preds, target_names=['bonafide', 'spoof']))

    #Plot the ROC curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr,tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    plt.show()

