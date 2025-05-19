# Detector Program

Program `detector.py` has all 3 subtasks.

To run the program, use a command like this one:

```sh
python detector.py No_entry/NoEntry15.bmp
```

where `No_entry/NoEntry15.bmp` is your image path. Please include the file extension with the image path like shown

By default, the program will run "subtask 3," which is the best detector. At the end of the file, you will find this:

Uncomment whichever subtask you need to run, and comment the other one. Run only one subtask at a time so you don't overwrite the outputted files.

Each subtask will print an F1 and TPR score of the image inputted and will write a file `detected.jpg`, which is the image with the bounding boxes and ground truth.

For subtask 2, it will also write two more files: `magnitude.jpg`, which contains the magnitude of the image inputted into the Hough transform, and `houghspace.jpg`, which is a 2D representation of the Hough space of the image.


