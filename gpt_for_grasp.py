#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
import actionlib
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from tocabi_msgs.msg import positionCommand
import cv2

from archimist_msgs.msg import ImgReqAction, ImgReqGoal

from openai import OpenAI
import numpy as np
import base64
import re
from datetime import datetime
import time
import json
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")
warnings.filterwarnings("ignore", category=FutureWarning, message="resume_download")


from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
model = load_model("/home/dyros/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/dyros/GroundingDINO/weights/groundingdino_swint_ogc.pth")


API_KEY='your_key'

img = None
dimg = None
current_joint = None
seq=None
joint_cmd = positionCommand()
joint_cmd.gravity = True
joint_cmd.relative = False
joint_cmd.traj_time = 1

class Chat:
    def __init__(self, 
                 model,
                 messages,
                 max_tokens=300):

        self.model = model
        self.max_tokens = max_tokens
        self.messages = messages
        self.client = OpenAI(api_key=API_KEY)
        # self.messages = [{"role": "system", 
        #     "content": "You are a helpful assistant that can plan household tasks. From now on, assume that you are a humanoid robot in a real environment and answer. When I give you an order, please answer with a small job unit prompt with verbs and nouns (1 or 2) to do this. If you need multiple prompts in succession to execute a command, you should list them in order using comma(,)"}]

    def ask(self, text, image_url=None, dimage_url=None):
        # self.messages.append({"role": "user", "content": text})
        # print((image_url))
        if self.model == "gpt-3.5-turbo" :
            self.messages.append({"role": "user", "content": text})

        else :
            self.messages.append({  "role": "user", 
                                    "content": [
                                    {
                                        "type": "text", 
                                        "text": text
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_url}",
                                            "detail": "auto"
                                        },
                                    },
                                    # {
                                    #     "type": "image_url",
                                    #     "image_url": {
                                    #         "url": f"data:image/jpeg;base64,{dimage_url}",
                                    #         "detail": "auto"
                                    #     },
                                    # },
                                ]})
        


        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
        )
        # print(self.messages)
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content




def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_image_(cv_image):
    if cv_image is None or cv_image.size == 0:
        raise ValueError("Empty or None image data provided.")
    
    _, buffer = cv2.imencode('.jpg', cv_image)

    return base64.b64encode(buffer).decode('utf-8')

def image_callback(goal_state, result):
    # print("image callback !!")
    global img
    global dimg
    global seq
    # print(goal_state)
    try:
        bridge = CvBridge()
        
        cv_image = bridge.imgmsg_to_cv2(result.image, "bgr8")
        img = cv_image
        # cv2.imwrite('/home/dyros/catkin_ws/src/tocabi/archimist/images/image.jpg', cv_image)
        cv2.imwrite(f'/home/dyros/catkin_ws/src/tocabi/archimist/images/image_{seq}.jpg', cv_image)
        # cv2.imwrite(f'/home/dyros/catkin_ws/src/tocabi/archimist/images/image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg', cv_image)


        # cv_dimage = bridge.imgmsg_to_cv2(result.dimage, "32FC1")
        # # Normalize the depth image to 0-255
        # cv_dimage = cv2.normalize(cv_dimage, None, 0, 255, cv2.NORM_MINMAX)
        # # Convert to 8-bit
        # cv_dimage = np.uint8(cv_dimage)
        # dimg = cv_dimage
        # cv2.imwrite('/home/dyros/catkin_ws/src/tocabi/archimist/images/dimage.jpg', cv_dimage)
        # rospy.loginfo("Image saved successfully")
    except Exception as e:
        rospy.logerr("Failed to convert image: %s" % e)

def jointstate_cb(data):
    global current_joint 
    global joint_cmd
    current_joint = data.position
    joint_cmd.position = [0 if abs(j) < 1e-4 else j for j in data.position]



def DINO(image, text):
    global seq
    IMAGE_PATH = f"/home/dyros/catkin_ws/src/tocabi/archimist/images/image_{seq}.jpg"
    text_without_robot = re.sub(r'\[Robot\] ', '', text)
    # --------부분과 그 밑 부분 제거
    TEXT_PROMPT = re.split(r'[-]{10,}', text_without_robot)[0].strip()
    BOX_TRESHOLD = 0.30
    TEXT_TRESHOLD = 0.30
    image_source, image = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    # print(boxes)
    # print(logits)
    # print(phrases)
    print("TEXT_PROMPT : ", TEXT_PROMPT)
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(f"/home/dyros/catkin_ws/src/tocabi/archimist/images/annotated_image_{seq}.jpg", annotated_frame)
        
def main():
    rospy.init_node('gpt2_node')

    # Subscriber
    rospy.Subscriber("/tocabi/jointstates", JointState, jointstate_cb)

    # Publisher
    joint_state_pub = rospy.Publisher('/tocabi/positioncommand', positionCommand, queue_size=10)

    # Action
    client = actionlib.SimpleActionClient('/imageRequestAction', ImgReqAction)
    client.wait_for_server()

    rate = rospy.Rate(20)  # 10 Hz

    global img
    global dimg
    global joint_cmd
    global current_joint
    global seq

    with open('/home/dyros/catkin_ws/src/tocabi/archimist/scripts/roleforgrasp.txt', "r") as file:
            role = file.read().splitlines()

    roleEye = [{
                "role": "system", 
                "content": role,
                }]
    EyeGPT = Chat(model="gpt-4o", messages=roleEye)

    audio_file= open("/home/dyros/catkin_ws/src/tocabi/archimist/videos/figure1.mp3", "rb")
    transcription = EyeGPT.client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    # ans = transcription.text
    # print(ans)
    ans = "[Robot]"

    action_list_pattern = r'<action to do: (.+)>'
    to_action_pattern = r'{\d+,\s*([^}]+)}'
    seq = 0
    action_list = []
    while not rospy.is_shutdown():
        try:

            if "[Robot]" in ans :
                action_to_do_match = re.search(action_list_pattern, ans)
                action_to_do_string = action_to_do_match.group(1) if action_to_do_match else ""
                action_list = re.findall(to_action_pattern, action_to_do_string) if action_to_do_string else []
                print("action_list:", action_list)

                if action_list:
                    ans = "[Command]"
                else:
                    print("[Human]", end=" ")
                    text = input()
                    ans = "[Human] " + text

            elif "[Human]" in ans :
                seq += 1
                img = None
                goal = ImgReqGoal(request=True)
                client.send_goal(goal, done_cb=image_callback)
                while((img is None)) : pass
                ans = EyeGPT.ask(ans, encode_image_(img))
                DINO(img,ans)
                text_without_robot = re.sub(r'\[Robot\] ', '', ans)
                # erase under '--------'
                txt = re.split(r'[-]{10,}', text_without_robot)[0].strip()
                res = EyeGPT.client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=txt,
                )
                # res.stream_to_file("/home/dyros/catkin_ws/src/tocabi/archimist/videos/output.mp3")
                print(ans)

            elif "[Command]" in ans :
                if action_list:
                    cur_action = action_list[0]
                    del action_list[0]
                    print("cur action : ", cur_action)
                else :
                    ans = "[Robot]"

                
        except EOFError:
            print("Control+D detected. Shutting down the node.")
            rospy.signal_shutdown('User requested shutdown via Control+D')
            break

        # print("------------------------------")
        # rate.sleep()


if __name__ == '__main__':
    main()