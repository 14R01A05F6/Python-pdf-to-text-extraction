#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 10:43:30 2019

@author: akshitbudhraja
"""
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import math
import numpy as np
from string import punctuation
import operator
import os
import re
from glob import glob
import cv2
#from parser import *

def apply_threshold(img, argument):
    switcher = {
            1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            18: cv2.adaptiveThreshold(cv2.medianBlur(img, 7), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
            19: cv2.adaptiveThreshold(cv2.medianBlur(img, 5), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
            20: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    }
    return switcher.get(argument, "Invalid method")

def process_image(img_path, method):
    # Read image using opencv
    img = cv2.imread(img_path)

    # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    # Create a directory for outputs
    output_path = os.path.join('out_result')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Apply threshold to get image with only black and white
    img = apply_threshold(img, method)
    # Save the filtered image in the output directory
    save_path = os.path.join(output_path + ".jpg")
    cv2.imwrite(save_path, img)


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_dates(path_to_pdf):
    extracted_text = list()
    print("Reading from file : " + str(path_to_pdf))
    pages = convert_from_path(path_to_pdf, 500)
    for page in pages:
        page.save('out.jpg', 'JPEG')
        process_image('out.jpg', 18)
        #correct_rotation('out.jpg')
        text = pytesseract.image_to_string(Image.open('out_result.jpg'))
        
        extracted_text.append(text)
    results = list()
    for text in extracted_text:
        text = text.split('\n')
        for line in text:
            line = line.split()
            for index, word in enumerate(line):
                if '/' in word:
                    word_ = word.split('/')
                    flag = 0
                    for w in word_:
                        if not is_number(w):
                            flag = -1
                            break
                    if flag == 0:
                        extracted_date = word
                        extracted_prev_text = list()
                        date_check = -1
                        date_index = -1
                        for i in range(index):
                            if 'date' in line[index - i - 1] or 'Date' in line[index - i - 1] or 'DATE' in line[index - i - 1]:
                                date_index = i
                                date_check = 0
                                break
                        if date_check == 0:
                            if date_index > 1:
                                print(str(line[index - date_index - 1:index+1]))
                                results.append(' '.join(line[index - date_index - 1:index+1]))
                            else:
                                if hasNumbers(line[index - date_index - 2]):
                                    print(str(line[index - date_index - 1:index+1]))
                                    results.append(' '.join(line[index - date_index - 1:index+1]))
                                else:
                                    print(str(line[index - date_index - 2:index+1]))
                                    results.append(' '.join(line[index - date_index - 2:index+1]))
    return results

def create_database(text):
    database = list()
    temp_text = text.split('\n')
    for ind, text_ in enumerate(temp_text):
        if ind == 0:
            continue
        if len(text_.split()) == 12:
            text_ = text_.split()
            temp_dict = dict()
            temp_dict['x1'] = float(text_[6])
            temp_dict['y1'] = float(text_[7])
            temp_dict['x2'] = float(text_[8]) + temp_dict['x1']
            temp_dict['y2'] = float(text_[9]) + temp_dict['y1']
            temp_dict['x'] = np.mean([temp_dict['x1'], temp_dict['x2']])
            temp_dict['y'] = np.mean([temp_dict['y1'], temp_dict['y2']])
            temp_dict['word'] = text_[11]
            database.append(temp_dict)
    return database
    
def extract_entity_index(database, entity_list):
    candidate_index = list()
    for ind, entry in enumerate(database):
            for word in entity_list:
                if word in entry['word']:
                    candidate_index.append(ind)
                    break
    return candidate_index

def extract_names_index(database):
    candidate_index = list()
    for ind, entry in enumerate(database):
        if 'Name' in entry['word'] or 'name' in entry['word'] or 'NAME' in entry['word']:
            candidate_index.append(ind)
    return candidate_index

def extract_nearby_words(database, candidate_index, stop_index = list(), upper_limit = 15, lower_limit = -15):
    candidate_values = list()
    for index in candidate_index:
        candidate_values_ = list()
        x = database[index]['x']
        y = database[index]['y']
        previous_word = ''
        min_distance = 0
        for ind, entry in enumerate(database):
            if ind in stop_index or ind == index:
                continue
            find_angle = math.degrees(math.atan2(entry['y'] - y, entry['x'] - x))
            if find_angle < upper_limit and find_angle > lower_limit and entry['word'] not in punctuation and sum([v not in punctuation for v in entry['word']]) !=0:
                entry_ = entry.copy()
                entry_['distance'] = math.sqrt((entry['y'] - y)**2 + (entry['x1'] - x)**2)
                entry_['candidate'] = database[index]
                candidate_values_.append(entry_)
            if find_angle >= (180 - upper_limit) or find_angle <= (-180 - lower_limit):
                distance = math.sqrt((entry['y'] - y)**2 + (entry['x2'] - x)**2)
                if distance < min_distance or min_distance == 0:
                    previous_word = entry['word']
                    min_distance = distance
        candidate_values_ = sorted(candidate_values_, key=operator.itemgetter('distance'))
        candidate_values.append([candidate_values_, previous_word])
    return candidate_values

def extract_entities(extracted_text, entity_list):
    results = list()
    for et_ind, extracted_text_ in enumerate(extracted_text):
        database = create_database(extracted_text_)
        candidate_index = extract_entity_index(database, entity_list)
        candidate_values = extract_nearby_words(database, candidate_index)
        for candidate_values_, previous_word in candidate_values:
            extracted_string_first = ''
            extracted_string_second = ''
            continuation_flag = False
            add_previous_word = False
            for cv_ind, cv in enumerate(candidate_values_):
                x1 = cv['x1']
                y1 = cv['y1']
                x2 = cv['x2']
                y2 = cv['y2']
                x = cv['x']
                y = cv['y']
                word = cv['word']    
                candidate_word = cv['candidate']['word']
                if cv_ind == 0:
                    extracted_string_first += candidate_word + ' '
                    candidate_x2 = x1
                    candidate_y1 = y1
                    if word == 'Name':
                        add_previous_word = False
                        continuation_flag = True
                if ((x1 - candidate_x2) >= 0 and (x1 - candidate_x2) <= 30 and abs(y1 - candidate_y1) <= 65) or (continuation_flag and (x1 - candidate_x2) >= 0 and (x1 - candidate_x2) <= 200 and abs(y1 - candidate_y1) <= 200):
                    if (x1 - candidate_x2) > 30:
                        continuation_flag = False
                    extracted_string_first += word + ' '
                    candidate_x2 = x2
                    candidate_y1 = y1
                else:
                    if ((x1 - candidate_x2) >= 0 and (x1 - candidate_x2) <= 30 and (y1 - candidate_y1) <= 100):
                        extracted_string_second += word + ' '
            
            extracted_string = extracted_string_first + extracted_string_second           
            extracted_string = extracted_string.lower()
            if 'Name' in extracted_string or 'name' in extracted_string or 'NAME' in extracted_string:
                continue
            if len(extracted_string.split()) > 2 and len(set(extracted_string).intersection(set(['>','<','!','#','$','*']))) == 0 and ('guardian' not in extracted_string and 'relationship' not in extracted_string) and not hasNumbers(extracted_string):
                if add_previous_word and len(previous_word) > 2:
                    results.append(previous_word + ' ' + extracted_string)
                else:
                    results.append(extracted_string)
    return results

def extract_names(path_to_pdf):
    extracted_text = list()
    print("Reading from file : " + str(path_to_pdf))
    pages = convert_from_path(path_to_pdf, 500)
    for page in pages:
        page.save('out.jpg', 'JPEG')
        process_image('out.jpg', 18)
        #correct_rotation('out.jpg')
        text = pytesseract.image_to_data(Image.open('out_result.jpg'))
        extracted_text.append(text)
    results = list()
    for et_ind, extracted_text_ in enumerate(extracted_text):
        database = create_database(extracted_text_)
        candidate_index = extract_names_index(database)
        candidate_values = extract_nearby_words(database, candidate_index)
        for candidate_values_, previous_word in candidate_values:
            extracted_string_first = ''
            extracted_string_second = ''
            continuation_flag = False
            add_previous_word = True
            for cv_ind, cv in enumerate(candidate_values_):
                x1 = cv['x1']
                y1 = cv['y1']
                x2 = cv['x2']
                y2 = cv['y2']
                x = cv['x']
                y = cv['y']
                word = cv['word']    
                candidate_word = cv['candidate']['word']
                if cv_ind == 0 or len(extracted_string_first.split()) == 1:
                    candidate_y1 = cv['candidate']['y1']
                    candidate_x2 = cv['candidate']['x2']
                    candidate_x1 = cv['candidate']['x1']
                if cv_ind == 0:
                    extracted_string_first += candidate_word + ' '
                    if word == 'of':
                        add_previous_word = False
                        continuation_flag = True
                if ((cv_ind == 0 or len(extracted_string_first.split()) == 1) and abs(y1 - candidate_y1) <= 85) or (((x1 - candidate_x2) >= 0 and (x1 - candidate_x2) <= 55 and abs(y1 - candidate_y1) <= 85) and cv_ind!=0) or (continuation_flag and (x1 - candidate_x2) >= 0 and (x1 - candidate_x2) <= 200 and abs(y1 - candidate_y1) <= 200):
                    if (x1 - candidate_x2) > 30:
                        continuation_flag = False
                    extracted_string_first += word + ' '
                    candidate_x2 = x2
                    candidate_y1 = y1
                    candidate_x1 = x1
                else:
                    if ((abs(x1 - candidate_x2) <= 250 or abs(x1 - candidate_x1) <= 250) and (y1 - candidate_y1) <= 150 and (y1 - candidate_y1) >=0):
                        extracted_string_second += word + ' '
            temp_string = extracted_string_second.split()
            flag___ = False
            for ind___, w in enumerate(temp_string):
                if hasNumbers(w):
                    flag___ = True
                    break
            if flag___ == True:
                extracted_string_second = extracted_string_second[:ind___]
            extracted_string = extracted_string_first + extracted_string_second
            extracted_string = extracted_string.lower()
            if len(extracted_string.split()) >= 2 and len(set(extracted_string).intersection(set(['>','<','#','$','*','°']))) == 0 and ('guardian' not in extracted_string and 'relationship' not in extracted_string) and not hasNumbers(extracted_string):
                if add_previous_word and len(previous_word) > 2:
                    results.append(previous_word + ' ' + extracted_string)
                else:
                    results.append(extracted_string)
    return results, extracted_text

#f = open('extracted_entities', 'w')
#dir_names = ['Metlife', 'TU_BR', 'UCA']
#dir_names = [os.path.join(os.getcwd(), d) for d in dir_names]
#for dir_name in dir_names:
#    dirs = sorted([os.path.join(dir_name, d) for d in os.listdir(dir_name) if re.match(r'[0-9][0-9][0-9][0-9][0-9][0-9]$', d)])
#    for i in range(len(dirs)):
#        fnames = sorted(glob(os.path.join(dirs[i], '*.' + 'pdf')))
#        for file_name in fnames:
#            results, extracted_text = extract_names(file_name)
#            dates = extract_dates(file_name)
#            d_results = extract_entities(extracted_text, ['Doctor', 'doctor', 'DOCTOR'])
#            pr_results = extract_entities(extracted_text, ['Provider', 'provider', 'PROVIDER'])
#            c_results = extract_entities(extracted_text, ['Clinic', 'clinic', 'CLINIC'])
#            f.write(file_name + '\n')
#            if dates:
#                for r in dates:
#                    f.write(str(r) + '\n')
#            if pr_results:
#                for r in pr_results:
#                    f.write(str(r) + '\n')
#            if d_results:
#                for r in d_results:
#                    f.write(str(r) + '\n')
#            if c_results:
#                for r in c_results:
#                    f.write(str(r) + '\n')
#            if results:
#                for r in results:
#                    f.write(str(r) + '\n')
#            f.write('\n\n\n')
#f.close()