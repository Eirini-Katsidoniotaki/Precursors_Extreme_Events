import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train models on FTLE/Fourier data")

    # Paths
    #parser.add_argument("--directory_FTLE", type=str, default="../data")
    parser.add_argument("--directory_modes", type=str, default="../data")
    #parser.add_argument("--directory_D", type=str, default="../data")
    parser.add_argument("--directory_data", type=str, default="../data")
    parser.add_argument("--directory_outputs", type=str, default="outputs")

    # Data
    #parser.add_argument("--IC_list", type=int, nargs="+", default=[13, 14, 15, 16, 17, 18])
    parser.add_argument("--downscale_factor", type=int, default=4)
    parser.add_argument("--DT_base", type=float, default=0.05)

    # Model choice
    parser.add_argument("--model_architecture", type=str, default="Informer", choices=["LSTM", "Transformer", "Informer"])
    parser.add_argument("--inp", type=str, default="FTLE", choices=["FTLE", "Fourier", "FTLE_Fourier"])

    # FTLE params
    parser.add_argument("--K",       type=int, default=100)
    parser.add_argument("--no_FTLE", type=int, default=1)

    # Filtering
    parser.add_argument("--cutoff",     type=float, default=0.0125)
    parser.add_argument("--sigma_diss", type=float, default=5)
    
    # Split data 
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size",  type=float, default=0.15)
    
    # Sequence creation
    parser.add_argument("--lookback", type=int, default=40, help="Lookback window in seconds")
    parser.add_argument("--tau",      type=int, default=10, help="Prediction horizon in seconds")
    parser.add_argument("--label",    type=int, default=20, help="Label length for Informer (seconds)")

    
    # Create batches
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    
    
    # Informer
    parser.add_argument("--inf_factor",   type=int,   default=7,    help="Informer attention factor")
    parser.add_argument("--inf_d_model",  type=int,   default=256,  help="Informer embedding dimension")
    parser.add_argument("--inf_n_heads",  type=int,   default=8,    help="Informer attention heads")
    parser.add_argument("--inf_e_layers", type=int,   default=3,    help="Number of Informer encoder layers")
    parser.add_argument("--inf_d_layers", type=int,   default=3,    help="Number of Informer decoder layers")
    parser.add_argument("--inf_d_ff",     type=int,   default=1024, help="Informer feedforward dimension")
    parser.add_argument("--dropout",      type=float, default=0.1,  help="dropout rate")
    # Informer additional options
    parser.add_argument("--inf_attn", type=str, default="prob", choices=["prob", "full"], help="Attention mechanism for Informer")
    parser.add_argument("--inf_embed", type=str, default="timeF", choices=["timeF", "fixed"], help="Embedding type for Informer")
    parser.add_argument("--inf_freq", type=str, default="m", choices=["s", "m", "h", "d"], help="Frequency for time features")
    parser.add_argument("--inf_activation", type=str, default="gelu", choices=["gelu", "relu"], help="Activation function")
    parser.add_argument("--inf_output_attention", action="store_true", help="If set, outputs attention maps")
    parser.add_argument("--no_inf_distil", action="store_true", help="Disable distilling in Informer encoder")
    parser.add_argument("--no_inf_mix", action="store_true", help="Disable mix attention")


    # Training
    parser.add_argument("--learning_rate",     type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay",      type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument("--alpha",             type=float, default=5.0,  help="Weighting factor for custom loss")
    parser.add_argument("--extreme_threshold", type=float, default=0.25, help="Threshold for extreme event classification")
    parser.add_argument("--bandwidth",         type=float, default=0.2,  help="Bandwidth for PDF lookup")

    
    # Training control
    parser.add_argument("--train_model", action="store_true",  help="If set, train the model. If not set, skip training.")
    parser.add_argument("--n_epochs", type=int, default=100,   help="Number of training epochs")
    parser.add_argument("--use_amp",  type=str, default="True", choices=["True", "False"], help="Enable mixed precision training (True/False)")
                    
    
    # Prediction control
    parser.add_argument("--predict_model", action="store_true", help="If set, run prediction/validation after training.")
                        
    # For metrics:
    # Extreme classification threshold
    parser.add_argument("--extreme_thres", type=float, default=0.17, help="Threshold for defining extreme events")
    
    return parser.parse_args()
