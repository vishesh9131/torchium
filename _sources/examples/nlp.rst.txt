Natural Language Processing Examples
====================================

This section provides comprehensive examples for NLP tasks using Torchium's specialized optimizers and loss functions.

Transformer Training
--------------------

Large Language Model Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium
   import math

   class TransformerModel(nn.Module):
       def __init__(self, vocab_size=50000, d_model=512, nhead=8, num_layers=6, max_len=1000):
           super().__init__()
           self.d_model = d_model
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
           self.transformer = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
               num_layers
           )
           self.classifier = nn.Linear(d_model, vocab_size)
           self.dropout = nn.Dropout(0.1)

       def forward(self, x, mask=None):
           seq_len = x.size(1)
           x = self.embedding(x) * math.sqrt(self.d_model)
           x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
           x = self.dropout(x)
           x = self.transformer(x, src_key_padding_mask=mask)
           return self.classifier(x)

   model = TransformerModel(vocab_size=50000, d_model=512, nhead=8, num_layers=6)

   # Use LAMB for large batch training
   optimizer = torchium.optimizers.LAMB(
       model.parameters(),
       lr=1e-3,
       betas=(0.9, 0.999),
       eps=1e-6,
       weight_decay=0.01,
       clamp_value=10.0
   )

   # Advanced NLP loss with label smoothing
   criterion = torchium.losses.LabelSmoothingLoss(
       num_classes=50000,
       smoothing=0.1
   )

   # Training loop with gradient clipping
   def train_transformer(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               output = model(batch.input_ids, batch.attention_mask)
               
               # Compute loss
               loss = criterion(output.view(-1, output.size(-1)), batch.labels.view(-1))
               
               # Backward pass
               loss.backward()
               
               # Gradient clipping
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Sequence-to-Sequence Models
---------------------------

Encoder-Decoder Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class Seq2SeqModel(nn.Module):
       def __init__(self, input_vocab_size, output_vocab_size, d_model=512, nhead=8, num_layers=6):
           super().__init__()
           self.d_model = d_model
           
           # Encoder
           self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
           self.encoder_pos_encoding = nn.Parameter(torch.randn(1000, d_model))
           self.encoder = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
               num_layers
           )
           
           # Decoder
           self.decoder_embedding = nn.Embedding(output_vocab_size, d_model)
           self.decoder_pos_encoding = nn.Parameter(torch.randn(1000, d_model))
           self.decoder = nn.TransformerDecoder(
               nn.TransformerDecoderLayer(d_model, nhead, batch_first=True),
               num_layers
           )
           
           # Output projection
           self.output_projection = nn.Linear(d_model, output_vocab_size)
           self.dropout = nn.Dropout(0.1)

       def forward(self, src, tgt, src_mask=None, tgt_mask=None):
           # Encoder
           src_seq_len = src.size(1)
           src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
           src_emb = src_emb + self.encoder_pos_encoding[:src_seq_len, :].unsqueeze(0)
           src_emb = self.dropout(src_emb)
           encoder_output = self.encoder(src_emb, src_key_padding_mask=src_mask)
           
           # Decoder
           tgt_seq_len = tgt.size(1)
           tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
           tgt_emb = tgt_emb + self.decoder_pos_encoding[:tgt_seq_len, :].unsqueeze(0)
           tgt_emb = self.dropout(tgt_emb)
           decoder_output = self.decoder(tgt_emb, encoder_output, 
                                        tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
           
           return self.output_projection(decoder_output)

   model = Seq2SeqModel(input_vocab_size=30000, output_vocab_size=30000)

   # Use NovoGrad for NLP tasks
   optimizer = torchium.optimizers.NovoGrad(
       model.parameters(),
       lr=1e-3,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=0.01,
       grad_averaging=True
   )

   # Combined loss for seq2seq
   class Seq2SeqLoss(nn.Module):
       def __init__(self, vocab_size=30000):
           super().__init__()
           self.ce_loss = torchium.losses.CrossEntropyLoss()
           self.label_smoothing = torchium.losses.LabelSmoothingLoss(
               num_classes=vocab_size, smoothing=0.1
           )

       def forward(self, pred, target):
           ce_loss = self.ce_loss(pred, target)
           smooth_loss = self.label_smoothing(pred, target)
           return 0.7 * ce_loss + 0.3 * smooth_loss

   criterion = Seq2SeqLoss()

   # Training loop for seq2seq
   def train_seq2seq(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
               
               # Compute loss
               loss = criterion(output.view(-1, output.size(-1)), batch.tgt.view(-1))
               
               # Backward pass
               loss.backward()
               
               # Gradient clipping
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Word Embeddings
---------------

Word2Vec Training
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class Word2VecModel(nn.Module):
       def __init__(self, vocab_size, embedding_dim=300):
           super().__init__()
           self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
           self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
           self.embedding_dim = embedding_dim

       def forward(self, target_words, context_words, negative_words):
           # Target embeddings
           target_emb = self.target_embeddings(target_words)
           
           # Context embeddings
           context_emb = self.context_embeddings(context_words)
           
           # Negative embeddings
           negative_emb = self.context_embeddings(negative_words)
           
           return target_emb, context_emb, negative_emb

   model = Word2VecModel(vocab_size=100000, embedding_dim=300)

   # Use SGD for word embeddings
   optimizer = torchium.optimizers.SGD(
       model.parameters(),
       lr=0.025,
       momentum=0.9
   )

   # Word2Vec specific loss
   criterion = torchium.losses.Word2VecLoss(
       vocab_size=100000,
       embedding_dim=300,
       negative_samples=5
   )

   # Training loop for Word2Vec
   def train_word2vec(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               target_emb, context_emb, negative_emb = model(
                   batch.target_words, batch.context_words, batch.negative_words
               )
               
               # Compute loss
               loss = criterion(target_emb, context_emb, negative_emb)
               
               # Backward pass
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

GloVe Training
~~~~~~~~~~~~~~

.. code-block:: python

   class GloVeModel(nn.Module):
       def __init__(self, vocab_size, embedding_dim=300):
           super().__init__()
           self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
           self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
           self.target_bias = nn.Embedding(vocab_size, 1)
           self.context_bias = nn.Embedding(vocab_size, 1)

       def forward(self, target_words, context_words, cooccurrence_counts):
           # Target embeddings
           target_emb = self.target_embeddings(target_words)
           target_bias = self.target_bias(target_words)
           
           # Context embeddings
           context_emb = self.context_embeddings(context_words)
           context_bias = self.context_bias(context_words)
           
           return target_emb, context_emb, target_bias, context_bias, cooccurrence_counts

   model = GloVeModel(vocab_size=100000, embedding_dim=300)

   # Use Adam for GloVe
   optimizer = torchium.optimizers.Adam(
       model.parameters(),
       lr=0.05
   )

   # GloVe specific loss
   criterion = torchium.losses.GloVeLoss(
       vocab_size=100000,
       embedding_dim=300,
       x_max=100.0,
       alpha=0.75
   )

   # Training loop for GloVe
   def train_glove(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               target_emb, context_emb, target_bias, context_bias, cooccurrence_counts = model(
                   batch.target_words, batch.context_words, batch.cooccurrence_counts
               )
               
               # Compute loss
               loss = criterion(target_emb, context_emb, target_bias, context_bias, cooccurrence_counts)
               
               # Backward pass
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Text Classification
-------------------

BERT-style Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BERTClassifier(nn.Module):
       def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12, num_classes=2):
           super().__init__()
           self.d_model = d_model
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.pos_encoding = nn.Parameter(torch.randn(512, d_model))
           self.segment_embedding = nn.Embedding(2, d_model)
           
           self.transformer = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
               num_layers
           )
           
           self.classifier = nn.Sequential(
               nn.Linear(d_model, d_model),
               nn.ReLU(),
               nn.Dropout(0.1),
               nn.Linear(d_model, num_classes)
           )

       def forward(self, input_ids, attention_mask, segment_ids):
           seq_len = input_ids.size(1)
           x = self.embedding(input_ids) * math.sqrt(self.d_model)
           x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
           x = x + self.segment_embedding(segment_ids)
           
           x = self.transformer(x, src_key_padding_mask=attention_mask)
           
           # Use [CLS] token for classification
           cls_output = x[:, 0, :]
           return self.classifier(cls_output)

   model = BERTClassifier(vocab_size=30000, num_classes=2)

   # Use AdamW for BERT-style training
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=2e-5,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=0.01
   )

   # Classification loss with label smoothing
   criterion = torchium.losses.LabelSmoothingLoss(
       num_classes=2,
       smoothing=0.1
   )

   # Training loop for BERT classification
   def train_bert_classifier(model, optimizer, criterion, dataloader, num_epochs=10):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               output = model(batch.input_ids, batch.attention_mask, batch.segment_ids)
               
               # Compute loss
               loss = criterion(output, batch.labels)
               
               # Backward pass
               loss.backward()
               
               # Gradient clipping
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Named Entity Recognition
------------------------

CRF-based NER
~~~~~~~~~~~~~

.. code-block:: python

   class NERModel(nn.Module):
       def __init__(self, vocab_size, num_tags, embedding_dim=128, hidden_dim=256):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, embedding_dim)
           self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
           self.classifier = nn.Linear(hidden_dim * 2, num_tags)

       def forward(self, input_ids, attention_mask):
           x = self.embedding(input_ids)
           x, _ = self.lstm(x)
           x = self.classifier(x)
           return x

   model = NERModel(vocab_size=30000, num_tags=9)  # BIO tagging scheme

   # Use AdamW for NER
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-3,
       weight_decay=1e-4
   )

   # CRF loss for NER
   criterion = torchium.losses.CRFLoss(
       num_tags=9,
       batch_first=True
   )

   # Training loop for NER
   def train_ner_model(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               output = model(batch.input_ids, batch.attention_mask)
               
               # Compute loss
               loss = criterion(output, batch.tags, batch.attention_mask)
               
               # Backward pass
               loss.backward()
               
               # Gradient clipping
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Text Generation
---------------

GPT-style Generation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class GPTModel(nn.Module):
       def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12, max_len=1024):
           super().__init__()
           self.d_model = d_model
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
           
           self.transformer = nn.TransformerDecoder(
               nn.TransformerDecoderLayer(d_model, nhead, batch_first=True),
               num_layers
           )
           
           self.classifier = nn.Linear(d_model, vocab_size)

       def forward(self, input_ids, attention_mask=None):
           seq_len = input_ids.size(1)
           x = self.embedding(input_ids) * math.sqrt(self.d_model)
           x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
           
           # Create causal mask
           tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
           
           x = self.transformer(x, x, tgt_mask=tgt_mask, memory_key_padding_mask=attention_mask)
           return self.classifier(x)

   model = GPTModel(vocab_size=50000, d_model=768, nhead=12, num_layers=12)

   # Use AdamW for GPT training
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-4,
       betas=(0.9, 0.95),
       eps=1e-8,
       weight_decay=0.1
   )

   # Perplexity loss for text generation
   criterion = torchium.losses.PerplexityLoss()

   # Training loop for GPT
   def train_gpt_model(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               output = model(batch.input_ids, batch.attention_mask)
               
               # Compute loss
               loss = criterion(output, batch.target_ids)
               
               # Backward pass
               loss.backward()
               
               # Gradient clipping
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Multi-Task NLP
--------------

Multi-Task Learning for NLP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskNLPModel(nn.Module):
       def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
           super().__init__()
           self.d_model = d_model
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
           
           self.transformer = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
               num_layers
           )
           
           # Task-specific heads
           self.classifier = nn.Linear(d_model, 2)      # Sentiment analysis
           self.ner_classifier = nn.Linear(d_model, 9)  # Named entity recognition
           self.qa_classifier = nn.Linear(d_model, 2)   # Question answering

       def forward(self, input_ids, attention_mask, task_type):
           seq_len = input_ids.size(1)
           x = self.embedding(input_ids) * math.sqrt(self.d_model)
           x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
           
           x = self.transformer(x, src_key_padding_mask=attention_mask)
           
           # Use [CLS] token for classification tasks
           cls_output = x[:, 0, :]
           
           if task_type == 'classification':
               return self.classifier(cls_output)
           elif task_type == 'ner':
               return self.ner_classifier(x)
           elif task_type == 'qa':
               return self.qa_classifier(cls_output)

   model = MultiTaskNLPModel(vocab_size=30000)

   # Use PCGrad for multi-task learning
   optimizer = torchium.optimizers.PCGrad(
       model.parameters(),
       lr=1e-3
   )

   # Multi-task loss with uncertainty weighting
   class MultiTaskNLPLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.uncertainty_loss = torchium.losses.UncertaintyWeightingLoss(num_tasks=3)
           self.cls_loss = torchium.losses.CrossEntropyLoss()
           self.ner_loss = torchium.losses.CRFLoss(num_tags=9, batch_first=True)
           self.qa_loss = torchium.losses.CrossEntropyLoss()

       def forward(self, cls_pred, ner_pred, qa_pred, cls_target, ner_target, qa_target, attention_mask):
           cls_loss = self.cls_loss(cls_pred, cls_target)
           ner_loss = self.ner_loss(ner_pred, ner_target, attention_mask)
           qa_loss = self.qa_loss(qa_pred, qa_target)
           
           return self.uncertainty_loss([cls_loss, ner_loss, qa_loss])

   criterion = MultiTaskNLPLoss()

   # Training loop for multi-task NLP
   def train_multitask_nlp(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass for each task
               cls_pred = model(batch.input_ids, batch.attention_mask, 'classification')
               ner_pred = model(batch.input_ids, batch.attention_mask, 'ner')
               qa_pred = model(batch.input_ids, batch.attention_mask, 'qa')
               
               # Compute loss
               loss = criterion(cls_pred, ner_pred, qa_pred, 
                               batch.cls_targets, batch.ner_targets, batch.qa_targets, 
                               batch.attention_mask)
               
               # Backward pass with gradient surgery
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

These examples demonstrate the power of Torchium's specialized optimizers and loss functions for various NLP tasks. Each example shows how to combine different components effectively for optimal performance in natural language processing.
