from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import main as detect
import os
import sys
import argparse
from PIL import Image
import pandas as pd


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result
    
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

if __name__ == '__main__':

    st.title('Image Segmentation Streamlit App')

    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str,
    #                     default='weights/yolov5s.pt', help='weights')
    # parser.add_argument('--source', type=str,
    #                     default='data/images', help='source')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--conf-thres', type=float,
    #                     default=0.35, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float,
    #                     default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='',
    #                     help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true',
    #                     help='display results')
    # parser.add_argument('--save-txt', action='store_true',
    #                     help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true',
    #                     help='save confidences in --save-txt labels')
    # parser.add_argument('--nosave', action='store_true',
    #                     help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int,
    #                     help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true',
    #                     help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true',
    #                     help='augmented inference')
    # parser.add_argument('--update', action='store_true',
    #                     help='update all models')
    # parser.add_argument('--project', default='runs/detect',
    #                     help='save results to project/name')
    # parser.add_argument('--name', default='exp',
    #                     help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true',
    #                     help='existing project/name ok, do not increment')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt = parser.parse_args()
    print(opt)
    
    model = ("yolo5s", "best")
    model_index = st.sidebar.selectbox("Select Model", range(
        len(model)), format_func=lambda x: model[x])

    if model_index == 0:
        opt.weights = f'weights/yolov5s.pt'
        print('when 0', opt)
    elif model_index == 1:
        opt.weights = f'weights/best.pt'
        print('when ', opt)
 
    source = ("Photo", "Video")
    source_index = st.sidebar.selectbox("Select File Type", range(
        len(source)), format_func=lambda x: source[x])
    
    preference = ("Time", "Calorie", "Rating")
    preference_index = st.sidebar.selectbox("Select Preference", range(
        len(preference)), format_func=lambda x: preference[x])
    prefdict = { "Time": "total_time", "Calorie": "nutrition","Rating": "rating" }
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Photo", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Resource Loading...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("Video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Resource Loading...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('Start Detection'):
            ingredidents = []
            detect(opt)
            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        if img == 'labels':
                            for txt in os.listdir(str(Path(f'{get_detection_folder()}') / img)):
                                file1 = open(str(Path(f'{get_detection_folder()}') / img / txt), 'r')
                                Lines = file1.readlines()
                                for line in Lines:
                                    x = line.split()
                                    ingred = " ".join(x[0:len(x)-5])
                                    ingredidents.append(ingred)
                        print('Ingredients ', set(ingredidents))
                        if not os.path.isdir(str(Path(f'{get_detection_folder()}') / img)):
                            st.image(str(Path(f'{get_detection_folder()}') / img))

                    st.balloons()
            else:
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid == 'labels':
                            for txt in os.listdir(str(Path(f'{get_detection_folder()}') / vid)):
                                file1 = open(str(Path(f'{get_detection_folder()}') / vid / txt), 'r')
                                Lines = file1.readlines()
                                for line in Lines:
                                    x = line.split()
                                    ingred = " ".join(x[0:len(x)-5])
                                    ingredidents.append(ingred)
                        print('Ingredients ', set(ingredidents))                       
                        if not os.path.isdir(str(Path(f'{get_detection_folder()}') / vid)):
                            st.video(str(Path(f'{get_detection_folder()}') / vid))
                    st.balloons()
            if len(ingredidents) > 0 :
                st.subheader('Ingredients Predicted')
                for i in set(ingredidents):
                    st.markdown("- " + i)
                st.subheader('Recipies Recommended' + '( Based on ' + preference[preference_index] + ' )')
                data = pd.read_csv("data/100_Recipes.csv") #path folder of the data file
                df = data.sort_values(prefdict[preference[preference_index]],ascending=True)
                st.write(df) #displays the table of data
