from facenet_pytorch import MTCNN

class FaceDetector:
    def __init__(self,device):
        self.mtcnn =MTCNN(
                image_size=224, margin=0, min_face_size=10,
                thresholds=[0.5, 0.6, 0.6], factor=0.709, post_process=False,
                device=device
            )
    
    def detect(self,img):
        if self.mtcnn.detect(img)[0] is not None:
            return True
        else:
            return False 