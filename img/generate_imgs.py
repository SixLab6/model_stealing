"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chromedriver_path = r'E:/AI_Python/Scripts/chromedriver'
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(executable_path=chromedriver_path, options=options)
driver.get('https://huggingface.co/spaces/google/sdxl')

def get_element_info(element, depth=0):
    attributes = driver.execute_script(
        'var items = {}; for (var index = 0; index < arguments[0].attributes.length; ++index) { '
        'items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value }; return items;',
        element
    )
    print(" " * depth * 4, element.tag_name, attributes)
    children = element.find_elements(By.XPATH, "./*")
    for child in children:
        get_element_info(child, depth + 1)
root_element = driver.find_element(By.XPATH, '/*')
get_element_info(root_element)
driver.quit()
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import base64
import urllib.parse
import requests
import os
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

cifar10_classes = ["tiger","elephant","shark","bicycle","rocket", "tank"]

generate_url="https://huggingface.co/spaces/google/sdxl"
def get_one_step_imgs(tep_url,prompt,epoch,num_class):
    chromedriver_path = r'E:/AI_Python/Scripts/chromedriver'
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(executable_path=chromedriver_path, options=options)
    driver.get(tep_url)
    time.sleep(15)
    flag = 0
    try:
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for index, iframe in enumerate(iframes):
            driver.switch_to.frame(iframe)
            inputs = driver.find_elements(By.TAG_NAME, "input")
            for input in inputs:
                for attribute in input.get_property('attributes'):
                    if attribute['name'] == 'placeholder' and attribute['value'] == 'Enter your prompt':
                        flag = 1
                        tep_input = input
                        break
                if 1 == flag:
                    tep_input.send_keys(prompt)
                    break
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                if button.text == 'Generate':
                    button.click()
                    print('Button clicked.')
                    time.sleep(20)
            imgs = driver.find_elements(By.TAG_NAME, "img")
            cnt=0
            for one_img in imgs:
                cnt=cnt+1
                image_url = one_img.get_attribute('src')
                if image_url.find('base64')!=-1:
                    base64_string = image_url.split("base64,")[-1]
                    base64_string = urllib.parse.unquote(base64_string)
                    while len(base64_string) % 4:
                        base64_string += '='
                    image_path = "./new_data/generated_"+cifar10_classes[num_class]+"/image_" +str(epoch)+ str(cnt) + ".jpg"
                    image_data = base64.b64decode(base64_string)
                    with open(image_path, 'wb') as file:
                        file.write(image_data)
            driver.switch_to.default_content()
    except Exception as e:
        print('Error:', e)
    finally:
        driver.quit()
import numpy as np

for num_class in range(len(cifar10_classes)):
    if num_class not in [2,3,4]:
        continue
    idx = 0
    data_prom = 'cifar100 training data, Just a single ' + cifar10_classes[
        num_class] + '; Realistic style; Clear background'
    mypath="./new_data/generated_"+cifar10_classes[num_class]
    if not os.path.exists(mypath):
        os.mkdir(mypath)
    """
    else:
        begin_idx=[]
        dirs = os.listdir(mypath)
        for dir in dirs:
            begin_idx.append(int(dir.split('_')[1][:-4]))
        if len(begin_idx)!=0:
            begin_idx = np.array(begin_idx)
            idx=begin_idx.max()/10
    """
    for i in range(200):
        get_one_step_imgs(generate_url, data_prom, i,num_class)