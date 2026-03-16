# CNN LSTM 
improvised version of https://github.com/pranoyr/cnn-lstm

# Getting Started
used vscode

### Try on your own dataset 

```
mkdir data
mkdir data/video_data
```
Put your video dataset inside data/video_data
It should be in this form --

```
data\video_data\
    reaches_for_screwdriver\
        clip1.mov ...
    no_interaction\
        clip1.mov
```

Generate Images from the Video dataset
```
./utils/generate_data.sh
```

## Train
Once you have created the dataset, start training ->
```
python main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes <num_classes>
```
but i used this instead (python main.py --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 0 --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/ --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes 2)

## Note 
* All the weights will be saved to the snapshots folder 
* To resume Training from any checkpoint, Use
```
--resume_path <path-to-model> 
```


## tensorboard visualization didn't work easily so it wasn't included


## References
* https://github.com/pranoyr/cnn-lstm
* https://github.com/kenshohara/video-classification-3d-cnn-pytorch
* https://github.com/HHTseng/video-classification

## License
This project is licensed under the MIT License 

