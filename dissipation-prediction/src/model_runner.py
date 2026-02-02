import os
import json
from datetime import datetime

from src.model.informer.models.model import Informer
from src.train import ModelTrain
from src.prediction import Predict
from src.utils_pdf import build_pdf_lookup
from src.losses import OutputWeightedMAE


def save_config(args, save_dir):
    """Save argparse arguments to a JSON file in save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")

    # Convert args Namespace → dict
    args_dict = vars(args)
    args_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(config_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"[INFO] Configuration saved to {config_path}")
    
    
    
    
def run_model(
    args,
    no_inputs,
    y_train_norm,
    train_loader,
    val_loader,
    test_loader,
    normalizer,
    device,
    DT,
    m,
    h,
    label_len
):
    preds, trues = None, None
    
    print("\n # ================================================== # ")
    print(f"      [INFO] {args.model_architecture} Architecture     ")
    print(" # ================================================== # ")

    # -------------------------------------------------
    # Experiment directories
    # -------------------------------------------------
    settings = (
        f"inp_{args.inp}_lookback_{args.lookback}_tau_{args.tau}"
        f"_label_{args.label}_batch_{args.batch_size}"
        f"_d_model_{args.inf_d_model}_nheads_{args.inf_n_heads}"
        f"_elayers_{args.inf_e_layers}_dlayers_{args.inf_d_layers}"
        f"_dff_{args.inf_d_ff}_drop_{args.dropout}"
    )

    save_dir = os.path.join(
        args.directory_outputs,
        f"tau_{args.tau}",
        f"{args.inp}"   # Alternatively, you can save the file as settings
    )

    model_save_dir = os.path.join(save_dir, "checkpoints")
    results_path   = os.path.join(save_dir, "results")
    model_name     = "model.pth"    #"InformerForecast.pth"

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    save_config(args, save_dir)

    # -------------------------------------------------
    # Build model
    # -------------------------------------------------
    model = Informer(
        enc_in=no_inputs, #X.shape[-1],
        dec_in=1,
        c_out=1,
        seq_len=m,
        label_len=label_len,
        out_len=h,
        factor=args.inf_factor,
        d_model=args.inf_d_model,
        n_heads=args.inf_n_heads,
        e_layers=args.inf_e_layers,
        d_layers=args.inf_d_layers,
        d_ff=args.inf_d_ff,
        dropout=args.dropout,
        attn=args.inf_attn,
        embed=args.inf_embed,
        freq=args.inf_freq,
        activation=args.inf_activation,
        output_attention=args.inf_output_attention,
        distil=not args.no_inf_distil,
        mix=not args.no_inf_mix,
        device=device
    ).to(device)

    #print("\n --- Model Architecture ----")
    #print(model)
    #print("===================================\n")

    # -------------------------------------------------
    # Trainer
    # -------------------------------------------------
    pdf_func = build_pdf_lookup(
        y_train_norm,
        bandwidth=args.bandwidth
    )

    trainer = ModelTrain(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        extreme_threshold=args.extreme_threshold,
        pdf_func=pdf_func
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    if args.train_model:
        print("\n # ============================================ # ")
        print(f" [INFO] Training {args.model_architecture} model ")
        print(" # ========================================== # \n")

        use_amp = args.use_amp.lower() == "true"

        trainer.train_informer(
            train_loader=train_loader,
            val_loader=val_loader,
            pred_len=h,
            label_len=label_len,
            n_epochs=args.n_epochs,
            save_dir=model_save_dir,
            model_name=model_name,
            use_amp=use_amp
        )
    else:
        print("\n # ============================= # ")
        print("     [INFO] The model is trained.    ")
        print(" # ============================= # ")

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    if args.predict_model:
        print("\n # ============================== # ")
        print("    [INFO] Prediction on test data   ")
        print(" # =============================== # ")

        predictor = Predict(
            model=model,
            device=device,
            test_loader=test_loader,
            save_dir=model_save_dir,
            model_name=model_name,
            results_path=results_path
        )

        preds, trues = predictor.run_informer(
            pred_len=h,
            label_len=label_len
        )

        # Inverse normalization
        _, preds = normalizer.inverse_transform(yn=preds)
        _, trues = normalizer.inverse_transform(yn=trues)
        
        print("\n # ============================ # ")
        print(f" [INFO] Plot the prediction .... ")
        print(" # ============================ # \n")

        k_list = [h - 1] #[0, int(0.5 * h) - 1, int(0.75 * h) - 1, h - 1]
        predictor.plot(k_list, DT)

    return preds, trues
