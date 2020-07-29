# trt_ssd
This is a tensorrt example for person detect.

## Begin
execuate below and you will get 'res.jpg'
```Shell
mkdir build&& cd build
cmake -D tensor_root=<tensorrt_root> -D OpenCV_DIR=<opencv_root> ..
make
./detect

```

