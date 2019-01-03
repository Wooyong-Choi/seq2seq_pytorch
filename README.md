# seq2seq_pytorch
A Seq2seq model for general task
Implementation of GRU based Encoder-Decoder with attention mechanism, and also stuff for preprocessing data, training model, and evaluating model.

We only support Pytorch 1.0.

## Features
- Framework Supporting data preprocessing
- Framework Supporting training and evaluating models
- GRU based Seq2seq network
- Attention mechanism (Luong, 2015)
- Beam search while decoding
- Attention visualization (To be updated)

## Requirements
```bash
pip install -r requirements.txt
```

## Quick start

Note that you'd better to see tutorial.ipynb in our repo.

### Step 1: Preprocess data
```bash
dataset = Dataset(
    src_file_path=SRC_FILE_PATH,
    tgt_file_path=TGT_FILE_PATH,
    max_src_len=MAX_SRC_LEN,
    max_tgt_len=MAX_TGT_LEN
)
```

### Step 2: Define model
```bash
model = Seq2seqModel(
    n_layers=NUM_LAYERS,
    input_size=INPUT_SIZE,
    emb_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    max_tgt_len=MAX_TGT_LEN,
    dropout_p=0.0,
    bi_encoder=True,
    device=device
)
```

### Step 3: Train model
```bash
trainer = Trainer(
    model=model,
    dataset=dataset,
    device=device,
    print_interval=1,
    plot_interval=-1,
    checkpoint_interval=10,
    expr_path=EXPR_PATH
)
trainer.train(num_epoch=NUM_EPOCH, batch_size=BATCH_SIZE)
```

### Step 4: Evaluate model
```bash
evaluator = Evaluator(dataset=dataset, model=model, device=device)
evaluator.loadModel(EXPR_PATH+'ep10.model')

pairs, attn_list = evaluator.evalModel(num=3, beam_size=5)
for p, attn in zip(pairs, attn_list):
    print('Input : ' + ' '.join(p[0]))
    print('Gen   : ' + ' '.join(p[1]))
    print()
```
