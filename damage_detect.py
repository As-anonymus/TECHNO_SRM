import torch
import streamlit as st
from PIL import Image
import cv2
from io import StringIO
import os
import shutil
from zipfile import ZipFile
from glob import glob



with st.sidebar:
    st.title("Vehicle Damage Detection")




# Function for model integration with streamlit 

def convert_df(df):
   '''
      For convering datframe to CSV
   '''
   return df.to_csv().encode('utf-8')



def image_dt(uploaded_file,model):
    '''
    Detcting damages using damage detction model on images.
    '''

    img1=model(os.path.join('data/images/',uploaded_file.name))
    df=img1.pandas().xyxy[0]
    img1=img1.save(f'data/images/output/{uploaded_file.name}')

    return df


""" We have used pre annoted dataset from Roboflow for creating model. The dataset consists of 10675 images with 17 classes like Front-Windscreen-Damage, Headlight-Damage, Major-Rear-Bumper-Dent, 
Rear-windscreen-Damage, RunningBoard-Dent, Sidemirror-Damage, Signlight-Damage, Taillight-Damage, bonnet-dent, doorouter-dent, fender-dent, front-bumper-dent, medium-Bodypanel-Dent, pillar-dent, 
quaterpanel-dent, rear-bumper-dent, roof-dent. """
    
def video_dt(uploaded_file,model):
    '''
    Detecting damages using damage detection model on videos.
    '''

    vid= cv2.VideoCapture(os.path.join("data", "videos", uploaded_file.name))     
    i=0
    while True:
        hasFrames,image = vid.read()
        if not hasFrames:
            break  # no more frames to read
        
        if i % 1 == 0:
            re = model(image[..., ::-1])
            if re is not None:
                re.save(f'data/images/output/{uploaded_file.name}')    
                image_path = os.path.join('result', f'im{i}.jpg')
                shutil.copy(os.path.join(max(glob("runs/detect/*/"), key=os.path.getmtime),"image0.jpg"), image_path)
                print(f"Result image saved to {image_path}")

            
        i += 1
        print(i)
        if i >= 1000:
            break  # limit to 1000 frames
        
    images = [img for img in os.listdir('result') if img.endswith(".jpg")]
    temp_vid = uploaded_file.name.split('.')[0] + '.webm'
    video = cv2.VideoWriter(os.path.join('vid', temp_vid), fourcc=cv2.VideoWriter_fourcc(*'vp80'), fps=5, frameSize=(1280,720), isColor=True)
    for i in images:
        im = cv2.imread(os.path.join('result', i))
        im = cv2.resize(im, (1280,720))
   
        video.write(im)
        os.remove(os.path.join('result', i))

    video.release()

    # using zipper to create zip file of video for downloading.

    old_path = os.getcwd()
    os.chdir('zipr')
    shutil.make_archive(uploaded_file.name.split('.')[0], 'zip', root_dir=os.path.join('vid', temp_vid))
    os.chdir(old_path)
    with ZipFile(os.path.join('zipr', uploaded_file.name.split('.')[0] + '.zip'), "w") as newzip:
        newzip.write(os.path.join('vid', temp_vid))

   

#-----Uplaoding Picture and video ------ # 




def upload(model):
    source  = ( "Image Detection" , "Video Detection", "Multiple Image Detection" )
    source_index  =  st.sidebar.selectbox ( " Select Input ", range (len(source)), format_func=lambda x: source[x])
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader("Upload Image" , type = [ 'png' , 'jpeg' , 'jpg' ])
        if uploaded_file is not None:
            is_valid = True
            with  st . spinner ( text = 'Resource loading...' ):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')

    elif source_index == 1:
        uploaded_file  =  st.sidebar.file_uploader ( " Upload Video" , type = [ 'mp4' ] )
        if uploaded_file is not None:
            is_valid = True
            with  st . spinner ( text = 'Resource loading...' ):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer()) # save video 
                    f.write(uploaded_file.read())      # save video 
                
        else:
            is_valid = False

    else:
        uploaded_files = st.sidebar.file_uploader("Upload Multiple Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files is not None:
            is_valid = True
            for uploaded_file in uploaded_files:
                with st.spinner('Loading image...'):
                    st.sidebar.image(uploaded_file)
                    picture = Image.open(uploaded_file)
                    picture.save(f'data/images/{uploaded_file.name}')

        else:
            is_valid = False




    if st.button("Detect"):

        if source_index==0:

            df=image_dt(uploaded_file,model)

            temp_img=uploaded_file.name.split('.')[0]+'.jpg'
            st.image(os.path.join(max(glob("runs/detect/*/"), key=os.path.getmtime),temp_img))
            st.table(df)
            csv = convert_df(df)

            st.download_button("Download CSV",csv,"file.csv","text/csv",key='download-csv'
            )
            with open(os.path.join(max(glob("runs/detect/*/"), key=os.path.getmtime),temp_img), "rb") as file:
                st.download_button("Download Image",data=file,file_name='img.jpg',mime="image/jpeg")
        
        else:
            with  st . spinner ( text = 'Detecting...' ):
            
                vid= video_dt(uploaded_file,model)
            
            temp_vid=uploaded_file.name.split('.')[0]+'.webm'
            temp_zip=uploaded_file.name.split('.')[0]+'.zip'
            video_file = open(os.path.join('vid',temp_vid), 'rb')
            video_bytes = video_file.read()
            

            st.video(video_bytes)
            with open(os.path.join('zipr',temp_zip), "rb") as fp:
                st.download_button("Download Video in Zip format",data=fp,file_name='result.zip',mime="application/zip")

pass



#Pages Layout 



def pg1():

    # Using pytorch hub inference of Yolov5 for loading model
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

    model = torch.hub.load('ultralytics/yolov5', 'custom', path = "model/dam_det.pt",force_reload=True)
    

    upload(model)
    pass



page_names_to_funcs = {
    "Vehicle Damage Detection":pg1,
   
}

selected_page = st.sidebar.radio(" ",page_names_to_funcs.keys())
page_names_to_funcs [selected_page]() 








