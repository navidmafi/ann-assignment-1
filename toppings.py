ckpt_path = "best.pt"
try:
    pass
    # checkpoint = torch.load(ckpt_path, map_location=TORCH_DEVICE, weights_only=False)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # start_epoch = checkpoint["epoch"] + 1
    # print(f"Resuming from epoch {start_epoch}")
except FileNotFoundError:
    pass


 # -----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # print(f"Saving model to {ckpt_path}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
        }, ckpt_path)



    output = model(torch.randn(1, 3, 28, 28).cuda())
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(f"figures/{id}_comp_graph", format="png")



    # table_tex = summary_df.style.hide(axis=1).format(escape="latex").set_table_styles([
    #     {'selector': 'toprule', 'props': ':hline;'},
    #     # {'selector': 'midrule', 'props': ':hline;'},
    #     {'selector': 'bottomrule', 'props': ':hline;'},
    # ], overwrite=True).to_latex(column_format='|c|c|', caption=title, label=id,position_float="centering")
    # table_tex = summary_df.to_latex(escape=True, caption=title, label=id, header=False, column_format='|c|c|')




# def serialize_optimizer(optimizer):
#     allowed_keys = {"lr", "momentum", "weight_decay", "betas", "eps"}
#     parts = []
#     for i, group in enumerate(optimizer.param_groups):
#         items = [f"{k}={v}" for k, v in group.items() if k in allowed_keys]
#         parts.append(f"Group {i}: " + ", ".join(items))
#     return "; ".join(parts)


    fig = doc.create(Figure(position='H'))
    fig.add_image(fig_path, width=NoEscape(width))
    fig.add_caption(f"{title} Loss/Accuracy graph")
    fig.append(NoEscape(f"\\label{{{label}_graph}}"))
