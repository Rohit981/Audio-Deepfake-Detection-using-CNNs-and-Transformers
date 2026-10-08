# import torch
# import torch.nn as nn
# from transformers import Wav2Vec2Model, WavLMModel

# class SSLFronendClassifier(nn.Module):
#     def __init__(self,
#                  model_type="wav2vec2",
#                  num_classes=1,
#                  freeze_frontend=True):
#         super().__init__()

#         print(f"[INFO] Intializing {model_type} frontend")
#         self.model_type = model_type

#         #Load the robust pre-trained foundation model wav2vec2 and wavlm
#         if self.model_type == 'wav2vec2':
#             self.frontend = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
#             self.feature_dim = self.frontend.config.hidden_size
#         elif self.model_type == 'wavlm':
#             self.frontend = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
#             self.feature_dim = self.frontend.config.hidden_size
#         else:
#             raise ValueError("Supported model types are 'wav2vec2' or 'wavlm'")

#         if freeze_frontend:
#             for parm in self.frontend.parameters():
#                 parm.requires_grad = False

#             self.frontend.eval()
#             print(f'[INFO] {model_type} frontend has been frozen')
#         else:
#             print(f"[WARNING] {model_type} frontend is trainable. High risk of overfitting to studio data!")

#         self.freeze_frontend = freeze_frontend

#         #Downstream classifier backend
#         self.classifier = nn.Sequential(
#             nn.Linear(self.feature_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def train(self, mode=True):
#         super(SSLFronendClassifier, self).train(mode)
#         # Force the frontend to stay in evaluation mode (turns off dropout/batchnorm updates) if frozen
#         if self.freeze_frontend:
#             self.frontend.eval()

#     def forward(self,x):

#         #Ensure input tensor is floating point
#         x = x.float()

#         #Extract features through the frozen foundation backend
#         if self.freeze_frontend:
#             with torch.no_grad():
#                 outputs = self.frontend(x)
#         else:
#             outputs = self.frontend(x)

#         #last_hidden_state shape: [Batch_Size, Time_Steps, Feature_Dim (768)]
#         features = outputs.last_hidden_state
        
#         # Global Temporal Pooling: Compress the time dimension down using Mean Pooling
#         pooled_features = torch.mean(features, dim=1) # Shape: [Batch_Size, Feature_Dim]
        
#         # Pass the robust embeddings into your classifier head
#         logits = self.classifier(pooled_features)
        
#         return logits.squeeze(-1) # Outputs a 1D tensor of logits [Batch_Size]