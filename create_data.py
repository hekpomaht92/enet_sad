import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
from collections import namedtuple


class DataCreator:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Label = namedtuple( 'Label' , [

        'name'        , # The identifier of this label.
                        # We use them to uniquely name a class

        'id'          , # An integer ID that is associated with this label.
                        # The IDs are used to represent the label in ground truth images
                        # An ID of -1 means that this label does not have an ID and thus
                        # is ignored when creating ground truth images (e.g. license plate).
                        # Do not modify these IDs, since exactly these IDs are expected by the
                        # evaluation server.

        'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                        # ground truth images with train IDs, using the tools provided in the
                        # 'preparation' folder. However, make sure to validate or submit results
                        # to our evaluation server using the regular IDs above!
                        # For trainIds, multiple labels might have the same ID. Then, these labels
                        # are mapped to the same class in the ground truth images. For the inverse
                        # mapping, we use the label that is defined first in the list below.
                        # For example, mapping all void-type classes to the same ID in training,
                        # might make sense for some approaches.
                        # Max value is 255!

        'category'    , # The name of the category that this label belongs to

        'categoryId'  , # The ID of this category. Used to create ground truth images
                        # on category level.

        'hasInstances', # Whether this label distinguishes between single instances or not

        'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                        # during evaluations or not

        'color'       , # The color of this label
        ] )


        #--------------------------------------------------------------------------------
        # A list of all labels
        #--------------------------------------------------------------------------------

        # Please adapt the train IDs as appropriate for your approach.
        # Note that you might want to ignore labels with ID 255 during training.
        # Further note that the current train IDs are only a suggestion. You can use whatever you like.
        # Make sure to provide your results using the original IDs and not the training IDs.
        # Note that many IDs are ignored in evaluation and thus you never need to predict these!

        self.labels = [
            #           name     id trainId      category  catId hasInstances ignoreInEval            color
            self.Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),
            self.Label(    's_w_d' , 200 ,     1 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ),
            self.Label(    's_y_d' , 204 ,     2 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),
            self.Label(  'ds_w_dn' , 213 ,     3 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ),
            self.Label(  'ds_y_dn' , 209 ,     4 ,   'dividing' ,   1 ,      False ,      False , (255, 0,   0) ),
            self.Label(  'sb_w_do' , 206 ,     5 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),
            self.Label(  'sb_y_do' , 207 ,     6 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),
            self.Label(    'b_w_g' , 201 ,     7 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ),
            self.Label(    'b_y_g' , 203 ,     8 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),
            self.Label(   'db_w_g' , 211 ,     9 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
            self.Label(   'db_y_g' , 208 ,    10 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
            self.Label(   'db_w_s' , 216 ,    11 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
            self.Label(    's_w_s' , 217 ,    12 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),
            self.Label(   'ds_w_s' , 215 ,    13 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
            self.Label(    's_w_c' , 218 ,    14 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
            self.Label(    's_y_c' , 219 ,    15 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
            self.Label(    's_w_p' , 210 ,    16 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),
            self.Label(    's_n_p' , 232 ,    17 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
            self.Label(   'c_wy_z' , 214 ,    18 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),
            self.Label(    'a_w_u' , 202 ,    19 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ),
            self.Label(    'a_w_t' , 220 ,    20 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
            self.Label(   'a_w_tl' , 221 ,    21 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
            self.Label(   'a_w_tr' , 222 ,    22 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
            self.Label(  'a_w_tlr' , 231 ,    23 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
            self.Label(    'a_w_l' , 224 ,    24 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
            self.Label(    'a_w_r' , 225 ,    25 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
            self.Label(   'a_w_lr' , 226 ,    26 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),
            self.Label(   'a_n_lu' , 230 ,    27 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
            self.Label(   'a_w_tu' , 228 ,    28 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ),
            self.Label(    'a_w_m' , 229 ,    29 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
            self.Label(    'a_y_t' , 233 ,    30 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
            self.Label(   'b_n_sr' , 205 ,    31 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
            self.Label(  'd_wy_za' , 212 ,    32 ,  'attention' ,   9 ,      False ,       True , (  0, 255, 255) ),
            self.Label(  'r_wy_np' , 227 ,    33 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),
            self.Label( 'vom_wy_n' , 223 ,    34 ,     'others' ,  11 ,      False ,       True , (128, 128,  64) ),
            self.Label(   'om_n_n' , 250 ,    35 ,     'others' ,  11 ,      False ,      False , (102,   0, 204) ),
            self.Label(    'noise' , 249 ,   255 ,    'ignored' , 255 ,      False ,       True , (  0, 153, 153) ),
            self.Label(  'ignored' , 255 ,   255 ,    'ignored' , 255 ,      False ,       True , (255, 255, 255) ),
        ]

        self.shp = (512, 512)
    
    def change_images(self, color_image, label_image):

        color_image = color_image[-1620:, :, :]
        label_image = label_image[-1620:, :, :]
        label_image_gray = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
    
        label_image_gray_filtered = np.zeros_like(label_image_gray)
        for brightness in [138, 42, 55, 30]:
            label_image_gray_filter = label_image_gray.copy()
            label_image_gray_filter[label_image_gray != brightness] = 0
            label_image_gray_filtered += label_image_gray_filter

        # plt.figure(1)
        # plt.imshow(label_image_gray_filters)
        # plt.figure(2)
        # plt.imshow(label_image)
        # plt.show()
        
        coef = np.zeros((1,1,2)).astype('float')
        coef[0][0][1] = label_image_gray.shape[0]/self.shp[1]
        coef[0][0][0] = label_image_gray.shape[1]/self.shp[0]

        _, contours, _ = cv2.findContours(label_image_gray_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_ = []
        for contour in contours:
            contours_.append(np.int32(contour / coef))
        
        # img1 = np.zeros((label_image_gray.shape[0], label_image_gray.shape[1], 3)).astype('uint8')
        # cv2.drawContours(img1, contours, -1, (0,255,0), 3)

        # img2 = np.zeros((self.shp[1], self.shp[0], 3)).astype('uint8')
        # cv2.drawContours(img2, contours_, -1, (0,255,0), 1)

        label_image_id = np.zeros((self.shp[1], self.shp[0])).astype('uint8')

        for contourid in range(len(contours_)):

            area = cv2.contourArea(contours_[contourid])
            if area < 10:
                continue
            
            img1 = np.zeros((label_image_gray.shape[0], label_image_gray.shape[1], 3)).astype('uint8')
            cv2.drawContours(img1, contours, contourid, (0,255,0), 1)

            # plt.figure(1)
            # plt.imshow(img1)
            # plt.show()

            label = label_image[contours[contourid][0][0][1]][contours[contourid][0][0][0]]
            if len([x for x, y in zip(label, [70, 130, 180]) if x == y]) == 3:
                cv2.fillConvexPoly(label_image_id, contours_[contourid], 1)
            elif len([x for x, y in zip(label, [0, 0, 142]) if x == y]) == 3:
                cv2.fillConvexPoly(label_image_id, contours_[contourid], 1)
            elif len([x for x, y in zip(label, [255, 0, 0]) if x == y]) == 3:
                cv2.fillConvexPoly(label_image_id, contours_[contourid], 1)
            elif len([x for x, y in zip(label, [220, 20, 60]) if x == y]) == 3:
                cv2.fillConvexPoly(label_image_id, contours_[contourid], 1)
            elif len([x for x, y in zip(label, [119, 11, 32]) if x == y]) == 3:
                cv2.fillConvexPoly(label_image_id, contours_[contourid], 1)

        # plt.figure(1)
        # plt.imshow(color_image)
        # plt.figure(2)
        # plt.imshow(label_image)
        # plt.figure(3)
        # plt.imshow(label_image_id*255)
        # plt.figure(4)
        # plt.imshow(img1)
        # plt.figure(5)
        # plt.imshow(thresh)
        # plt.figure(6)
        # plt.imshow(img2)
        # plt.show()

        color_image = cv2.resize(color_image, self.shp, interpolation=cv2.INTER_AREA)
        label_image = cv2.resize(label_image, self.shp)

        return color_image, label_image, label_image_id
        

if __name__ == '__main__':

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    data_creator = DataCreator()
    ROOT_FOLDER = 'D:\\datasets\\Apollo'
    RESIZER_ROOT_FOLDER = 'D:\\datasets\\resized\\Apollo_512x512'
    # ROAD_FOLDERS = os.listdir(ROOT_FOLDER)
    # ROAD_FOLDERS.sort()
    ROAD_FOLDERS = ['road04'] #['road02', 'road03', 'road04']
    
    for road_folder in ROAD_FOLDERS: # read road folders

        label_folder = os.path.join(road_folder, 'Label')
        color_folder = os.path.join(road_folder, 'ColorImage')
        record_folders = os.listdir(os.path.join(ROOT_FOLDER, label_folder))
        record_folders.sort()
        # record_folders = ['Record043', 'Record044', 'Record045', 'Record046', 'Record047', 'Record048']
        
        for record_folder in record_folders: # read record folders0

            camera_folders = [os.path.join(record_folder, 'Camera 5'),
                              os.path.join(record_folder, 'Camera 6')]

            for camera_folder in camera_folders: # read camera folders

                list_image = os.listdir(os.path.join(ROOT_FOLDER, color_folder, camera_folder))
                list_image.sort()

                if not os.path.exists(os.path.join(RESIZER_ROOT_FOLDER, 'videos')):
                        os.makedirs(os.path.join(RESIZER_ROOT_FOLDER, 'videos'))

                video_color = cv2.VideoWriter(os.path.join(RESIZER_ROOT_FOLDER, 'videos', '{}_{}_{}_color.avi'\
                    .format(road_folder, camera_folder.split('\\')[0], camera_folder.split('\\')[0])),
                        fourcc, 20.0, data_creator.shp)
                video_label = cv2.VideoWriter(os.path.join(RESIZER_ROOT_FOLDER, 'videos', '{}_{}_{}_label.avi'\
                    .format(road_folder, camera_folder.split('\\')[0], camera_folder.split('\\')[0])),
                        fourcc, 20.0, data_creator.shp)
                video_labelid = cv2.VideoWriter(os.path.join(RESIZER_ROOT_FOLDER, 'videos', '{}_{}_{}_labelid.avi'\
                    .format(road_folder, camera_folder.split('\\')[0], camera_folder.split('\\')[0])),
                        fourcc, 20.0, data_creator.shp)

                for counter, image in enumerate(list_image): # read images
                    
                    print(counter)
                    image = image[:-4]

                    try:
                        color_image = cv2.imread(os.path.join(ROOT_FOLDER, color_folder,
                                                    camera_folder, image+'.jpg'))
                        label_image = cv2.imread(os.path.join(ROOT_FOLDER, label_folder,
                                                    camera_folder, image+'_bin.png'))

                        # color_image = cv2.imread('/media/hdd/apolloScapes/road03/ColorImage/Record001/Camera 5/171206_025744607_Camera_5.jpg')
                        # label_image = cv2.imread('/media/hdd/apolloScapes/road03/Label/Record001/Camera 5/171206_025744607_Camera_5_bin.png')
                        
                        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

                    except Exception as e:
                       print(e)
                       continue

                    color_image_resized, label_image_resized, label_image_id = \
                        data_creator.change_images(color_image, label_image)
                    
                    out_color_path = os.path.join(RESIZER_ROOT_FOLDER, color_folder, '{}_{}'.format(camera_folder[:-2], camera_folder[-1]))
                    out_label_path = os.path.join(RESIZER_ROOT_FOLDER, label_folder, '{}_{}'.format(camera_folder[:-2], camera_folder[-1]))

                    if not os.path.exists(out_color_path):
                        os.makedirs(out_color_path)
                    if not os.path.exists(out_label_path):
                        os.makedirs(out_label_path)

                    cv2.imwrite(os.path.join(out_color_path,'{0:05}.jpg'.format(counter)), color_image_resized)
                    cv2.imwrite(os.path.join(out_label_path,'{0:05}.png'.format(counter)), label_image_id)

                    label_image_id_video = np.concatenate((np.expand_dims(label_image_id, axis=2), 
                                                           np.expand_dims(label_image_id, axis=2),
                                                           np.expand_dims(label_image_id, axis=2)), axis=-1)
                    video_color.write(color_image_resized)
                    video_label.write(label_image_resized)
                    video_labelid.write(label_image_id_video*255)

                    with open(os.path.join('lists', 'image_list_{}.txt'.format(road_folder)), 'a', encoding='utf8') as f:
                        f.write(os.path.join(out_color_path,'{0:05}.jpg'.format(counter)) + ' ' +
                                os.path.join(out_label_path,'{0:05}.png'.format(counter)) + '\n')
                
                video_color.release()
                video_label.release()
                video_labelid.release()
    
        print('{} complite'.format(road_folder))

