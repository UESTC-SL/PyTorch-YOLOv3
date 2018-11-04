import argparse
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils.utils as tools
from models import Darknet
from utils.datasets import ImageFolder


DISPLAY_WIDTH = 1024
NUM_CLASSES = 80
OPT = None
SAVE_RESULTS = False
MAX_TRACK_FRAME = 60
CLASSES = {
    'car': (255, 79, 120),
    'person': (30, 84, 234),
    # 'truck': (32, 247, 135),
    'motorbike': (255, 216, 0)
}


class DrawElements:
    def __init__(self):
        self.drawing = False
        self.x = -1
        self.y = -1
        self.radius = 60


def check_overlap(target1, target2):
    """Check if bounding boxes of target1 and target2 are overlapped
    """
    if target1[2] <= target2[0] or target2[2] <= target1[0]:
        return False
    if target1[3] <= target2[1] or target2[3] <= target1[1]:
        return False
    return True


def get_overlap(target1, target2):
    """Overlap ratio = intersection area / minimal bounding box of {target1, target2}
    """
    area1 = (target1[2] - target2[0]) * (target1[3] - target2[1])
    area2 = (target2[2] - target1[0]) * (target2[3] - target1[1])
    inner = min(area1, area2)
    outter = max(area1, area2)
    return inner/outter


def track_anchorpoint(target):
    """Calculate anchor point for drawing the track line
    """
    return [(target[0]+target[2])//2, int(target[1]*0.25 + target[3]*0.75)]


def running_mean(x, N):
    """Fast moving average.
    Returns array, with size of x-N+1
    """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def smooth(y, size):
    """Mean smooth by convolution.
    Returns array, with size of max(y, size)
    """
    box = np.ones(size)/size
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def detect(model, sample):
    prev_time = time.time()
    detections = model(sample)
    detections = tools.non_max_suppression(detections, NUM_CLASSES, OPT.conf_thres, OPT.nms_thres)
    current_time = time.time()
    inference_time = current_time - prev_time
    return detections, inference_time


def detect_image(model, device, classes, window_name='Detections'):
    model.to(device)
    model.eval()
    dataloader = DataLoader(ImageFolder(OPT.image_folder, img_size=OPT.img_size),
                            batch_size=OPT.batch_size, shuffle=False, num_workers=OPT.n_cpu)
    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index
    print('\nPerforming object detection:')
    with torch.no_grad():
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = input_imgs.to(device)
            # Get detections
            detections, inference_time = detect(model, input_imgs)
            print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))
            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)
    # Bounding-box colors
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(NUM_CLASSES)]
    print('\nSaving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))
        img = cv2.imread(path)
        img = tools.resize(img, DISPLAY_WIDTH, None)
        height, width = img.shape[:2]
        # The amount of padding that was added
        if height <= width:
            pad_x = 0
            pad_y = (width - height) * OPT.img_size // width
        else:
            pad_x = (height - width) * OPT.img_size // height
            pad_y = 0
        # Image height and width after padding is removed
        unpad_h = OPT.img_size - pad_y
        unpad_w = OPT.img_size - pad_x
        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                # Rescale coordinates to original dimensions
                box_h = (y2 - y1) * height / unpad_h
                box_w = (x2 - x1) * width / unpad_w
                y1 = (y1 - pad_y // 2) * height / unpad_h
                x1 = (x1 - pad_x // 2) * width / unpad_w
                color = colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                cv2.rectangle(img, (x1, y1), (x1+box_w, y1+box_h), color, thickness=3)
                cv2.rectangle(img, (x1-2, y1-20), (x1+80, y1), color, thickness=-1)
                cv2.putText(img, classes[int(cls_pred)], (x1+3, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # Save generated image with detections
        cv2.imshow(window_name, img)
        # cv2.waitKey(0)
        # quit()
        cv2.waitKey(200)
        cv2.imwrite('output/%d.png' % (img_i), img)


def detect_video(model, device, classes, window_name='Detections', verbose=False):
    model.to(device)
    model.eval()
    # cap = cv2.VideoCapture(OPT.video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # ret, first_frame = cap.read()
    start_frame = 800
    first_frame = cv2.imread('../data/full_img/{}.jpg'.format(start_frame))
    # if not ret:
    #     raise Exception('Reading frames from {} failed.'.format(OPT.video_path))
    first_frame = tools.resize(first_frame, DISPLAY_WIDTH, None)
    height, width = first_frame.shape[:2]
    cv2.imshow(window_name, first_frame)
    draw_area = True
    elements = DrawElements()

    def draw_by_mouse(event, x, y, flags, param):
        """Draw detect area with mouse
        """
        nonlocal elements
        if event == cv2.EVENT_LBUTTONDOWN:
            elements.drawing = True
            elements.x, elements.y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            elements.x, elements.y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            elements.drawing = False
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_by_mouse)
    # Pause at first frame, and draw detect area
    #TODO: detect area should be better defined
    frame_area = np.zeros((height, width), dtype=np.float)
    while draw_area:
        if elements.drawing:
            # Draw new circle on `frame_area`
            cv2.circle(frame_area, (elements.x, elements.y), elements.radius, (1), -1)
            first_frame[:,:,2] = np.minimum(first_frame[:,:,2] + 128 * frame_area, 255)
        frame_show = np.copy(first_frame)
        cv2.putText(frame_show, "Draw road area with mouse. Confirm with SPACE.", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame_show, "Draw road area with mouse. Confirm with SPACE.", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(frame_show, "Use \'=\' and \'-\' to adjust painter size.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame_show, "Use \'=\' and \'-\' to adjust painter size.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(frame_show, "Press ESC to skip.", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame_show, "Press ESC to skip.", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        # Draw a circle that follows cursor
        cv2.circle(frame_show, (elements.x, elements.y), elements.radius, (0, 0, 220), -1)
        cv2.putText(frame_show, '{},{}'.format(elements.x, elements.y), (elements.x + 10, elements.y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow(window_name, frame_show)

        key = cv2.waitKey(5)
        if key == 27:
            # 'Esc' key. Accept the entire image as detect area
            frame_area = np.ones((height, width), dtype=np.float)
            draw_area = False
            break
        elif key == 32:
            # 'Space' key. Done drawing detect area and save it to file
            np.save('DetectArea.npy', frame_area)
            draw_area = False
            break
        elif key == 61:
            # '=' key. Larger cursor
            elements.radius += 2
            if elements.radius > 300:
                elements.radius = 300
        elif key == 45:
            # '-' key. Smaller cursor
            elements.radius -= 2
            if elements.radius < 2:
                elements.radius = 2
        elif key == 108:
            # 'L' key. Load saved detect area from file
            frame_area = np.load('DetectArea.npy')
            draw_area = False
            break

    car_detections = []     # for storing the lastest targets
    track_line = []         # for storing the original track line coordinates
    direction = None
    skip_frame = 4
    frame_count = 1 + start_frame
    # while cap.isOpened():
    while True:
        begin_time = time.time()
        # ret, frame = cap.read()
        frame = cv2.imread('../data/full_img/{}.jpg'.format(frame_count))
        if frame is None:
            break
        frame = tools.resize(frame, DISPLAY_WIDTH, None)
        frame_show = np.copy(frame)
        # Apply detect area to the frame
        frame[:,:,0] = np.multiply(frame[:,:,0], frame_area)
        frame[:,:,1] = np.multiply(frame[:,:,1], frame_area)
        frame[:,:,2] = np.multiply(frame[:,:,2], frame_area)
        pad = np.abs(height - width) // 2
        # resize to [416,416] as network input
        #TODO: [Optimize] scale first, pad later
        if height <= width:
            input_img = 255. * np.ones((width, width, 3), dtype=np.float)
            input_img[pad:height+pad,:,:] = np.copy(frame)
        else:
            input_img = 255. * np.ones((height, height, 3), dtype=np.float)
            input_img[:,pad:width+pad,:] = np.copy(frame)
        input_img *= 1./255.
        input_img = tools.resize(input_img, OPT.img_size, OPT.img_size)
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
        input_img = input_img.view(1, 3, OPT.img_size, OPT.img_size)
        input_img = input_img.to(device)
        # Detect objects with model
        detections, inference_time = detect(model, input_img)
        result = detections[0]
        # Get ready for resizing back
        if height <= width:
            pad_x = 0
            pad_y = (width - height) * OPT.img_size // width
        else:
            pad_x = (height - width) * OPT.img_size // height
            pad_y = 0
        unpad_h = OPT.img_size - pad_y
        unpad_w = OPT.img_size - pad_x
        # a group may contain multiple targets
        _group = []
        if result is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in result:
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                conf, cls_conf, cls_pred = conf.item(), cls_conf.item(), cls_pred.item()
                _cls = classes[int(cls_pred)]
                # scale bounding box to original image size
                box_h = (y2 - y1) * height // unpad_h
                box_w = (x2 - x1) * width // unpad_w
                y1 = (y1 - pad_y // 2) * height // unpad_h
                x1 = (x1 - pad_x // 2) * width // unpad_w
                x2 = x1 + box_w
                y2 = y1 + box_h
                #TODO: store `cls_conf` and `cls_pred` in the target
                _target = [x1, y1, x2, y2]
                if _cls in CLASSES:
                    if verbose:
                        print('\t+ Label: %s, Conf: %.5f' % (_cls, cls_conf))
                    # Only track cars
                    if _cls == 'car':
                        if len(car_detections) > 0:
                            # print(car_detections)
                            found_previous = False
                            # search for the same target that has previously appeared,
                            # starting from the newest ones
                            for group in reversed(car_detections):
                                if len(group) > 0:
                                    for target in group:
                                        # check each target in the group
                                        #TODO: need modification for multi-target tracking
                                        # print('1: {}'.format(target))
                                        # print('2: {}'.format(_target))
                                        if check_overlap(target, _target):
                                            # if two bounding boxes are overlapped,
                                            # consider them as the same target
                                            _overlap = get_overlap(target, _target)
                                            # print(_overlap)
                                            if _overlap > 0.8:
                                                found_previous = True
                                                track_line.append(track_anchorpoint(_target))
                                                break
                                # find target, terminate searching
                                if found_previous:
                                    break
                            _group.append(_target)
                    elif _cls == 'person':
                        # Do not draw that thingy mistaken as person
                        #TODO: allow disabling certain objects by hand
                        if check_overlap([264, 339, 317, 137], _target):
                            if get_overlap([264, 339, 317, 137], _target) > 0.4:
                                continue
                    # Draw bounding box for each object
                    color = CLASSES[_cls]
                    cv2.rectangle(frame_show, (x1, y1), (x1+box_w, y1+box_h), color, thickness=3)
                    cv2.rectangle(frame_show, (x1-2, y1-20), (x1+80, y1), color, thickness=-1)
                    cv2.putText(frame_show, _cls, (x1+3, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # if target is gone, clear the track line coordinates
        _count_empty = 0
        for g in car_detections:
            if len(g) == 0:
                _count_empty += 1
        if _count_empty > 30:
            track_line = []
            direction = None
            # track_line_smooth = []
        car_detections.append(_group)
        # print(car_detections)
        # delete the oldest target if length of `car_detections` becomes too large
        if len(car_detections) > MAX_TRACK_FRAME:
            car_detections.pop(0)
        # print(track_line)
        # smooth the track line by moving average
        #TODO: save the smoothed line coordinates that are already calculated
        if len(track_line) >= 9:
            track_line_np = np.array(track_line, dtype=np.float32)
            # print(track_line)
            # print(running_mean(track_line_np[:,0], 9))
            track_line_smooth = np.vstack((running_mean(track_line_np[:,0], 9), running_mean(track_line_np[:,1], 9)))
            track_line_smooth = np.transpose(track_line_smooth)
            # print(track_line_smooth)
            if track_line_smooth.shape[0] > 1:
                cv2.polylines(frame_show, np.array([track_line_smooth], dtype=np.int32), False, (0, 255, 255), 3)
                # cv2.polylines(frame_show, np.int32(track_line_smooth), False, (0, 255, 255), 3)
            # Check if the car is entering the parking lot
            if direction is None and track_line[0][0] < 400 and track_line[0][1] < 500 \
                and (track_line[-1][0] - track_line[0][0])**2 \
                + (track_line[-1][1] - track_line[0][1])**2 > 160000:
                direction = 0
            if direction == 0:
                cv2.putText(frame_show, "VEHICLE IN", (width//2-200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2., (0, 0, 0), 7)
                cv2.putText(frame_show, "VEHICLE IN", (width//2-200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2., (20, 20, 250), 3)
        run_time = time.time() - begin_time
        cv2.putText(frame_show, "FPS: {0:.1f}".format(1/run_time), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame_show, "FPS: {0:.1f}".format(1/run_time), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.imshow(window_name, frame_show)
        key = cv2.waitKey(1)
        # cv2.waitKey(0)
        # quit()
        if SAVE_RESULTS:
            cv2.imwrite('output/%d.png' % (frame_count), frame_show)
        if key == 27:
            break
        frame_count += 1
        #* Skip frame if necessary
        # while skip_frame > 0:
        #     ret, _ = cap.read()
        #     if not ret:
        #         break
        #     frame_count += 1
        #     skip_frame -= 1


def main():
    device = torch.device("cuda" if OPT.use_cuda else "cpu")
    cudnn.benchmark = True

    classes = tools.load_classes(OPT.class_path) # Extracts class labels from file

    model = Darknet(OPT.config_path, img_size=OPT.img_size)
    model.load_weights(OPT.weights_path)

    if OPT.source == 'image':
        detect_image(model, device, classes)
    elif OPT.source == 'video':
        detect_video(model, device, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='video', help='image or video or camera')
    parser.add_argument('--image_folder', type=str, default='data/images', help='path to dataset')
    parser.add_argument('--video_path', type=str, default='data/videos/stream.mp4', help='path to video file')
    parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    OPT = parser.parse_args()
    OPT.use_cuda = OPT.use_cuda and torch.cuda.is_available()

    os.makedirs('output', exist_ok=True)

    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}\n".format(torch.version.cuda))

    # Print all arguments.
    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("  " + "-" * 55)
    for arg in vars(OPT):
        arg_str = str(arg)
        var_str = str(getattr(OPT, arg))
        type_str = str(type(getattr(OPT, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)
    main()
