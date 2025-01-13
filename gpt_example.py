#!/usr/bin/env python
import cv2
from openai import OpenAI
import base64

API_KEY='your_key'

img = None

class Chat:
    def __init__(self, 
                 model,
                 messages,
                 max_tokens=300):

        self.model = model
        self.max_tokens = max_tokens
        self.messages = messages
        self.client = OpenAI(api_key=API_KEY)

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
                                ]})
        


        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            # max_tokens=self.max_tokens,
            # stream=True,
        )
        # print(self.messages)
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
        # for chunk in response:
        #     if chunk.choices[0].delta.content is not None:
        #         print(chunk.choices[0].delta.content, end="")
    
def encode_image(cv_image):
    if cv_image is None or cv_image.size == 0:
        raise ValueError("Empty or None image data provided.")
    _, buffer = cv2.imencode('.jpg', cv_image)
    return base64.b64encode(buffer).decode('utf-8')
        
def main():
    img = cv2.imread("/path/to/image")

    role   = [{
                "role": "system", 
                "content": "You are helpful AI assistant."
                }]
    # GPT = Chat(model="gpt-3.5-turbo", messages=role)
    GPT = Chat(model="gpt-4o", messages=role)

    while(True):
        try:
            print("---")
            text = input()
            # ans = GPT.ask(text)                   # only text
            ans = GPT.ask(text, encode_image(img)) # with image
            print(ans)
        except EOFError:
            print("Control+D detected. Shutting down the node.")
            break

if __name__ == '__main__':
    main()
