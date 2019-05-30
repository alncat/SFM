# SFM
First of all, you should prepare a dataset as it is detailed in https://github.com/tinghuiz/SfMLearner .

To train a model, run the command below,

python train_uni.py --dataset_dir=./kitti_raw_eigen/ --checkpoint_dir=./check/ --img_width=416 --img_height=128 --batch_size=8 --learning_rate=0.0001 --smooth_weight=0.01 --explain_reg_weight 0.01

To evaluate a model, run the command below to generate depth prediction

python test_kitti_depth_con.py --dataset_dir your_dataset_dir --output_dir pred/ --ckpt_file check/your_model_name

then compare with ground truth

python kitti_eval/eval_depth.py --kitti_dir=your_dataset_dir --pred_file=pred/your_prediction

You can download the best checkpoint so far at 

https://drive.google.com/file/d/1jljDcIiSbZcBmkIj6mN25Ft5Y5h8lu0r/view?usp=sharing

The implementation of warping process is located at utils.py, #bilinear_project# function.
The implementation of our network is in nets.py, #disp_aspp_u_dense# function
