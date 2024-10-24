import cv2
import numpy as np
import matplotlib.pyplot as plt



def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % \
        dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

# load images and downscale them for faster processing
imgL = cv2.pyrDown( cv2.imread('imageL.png') )  
imgR = cv2.pyrDown( cv2.imread('imageR.png') )

# apply denoising algorithm
imgL = cv2.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 21)
imgR = cv2.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 21)

imgL_bw = cv2.blur(cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY),(5,5))
imgR_bw = cv2.blur(cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY),(5,5))

plt.imshow(imgL)
# plt.imshow(imgR)
plt.show()



#Create disparity map
block_size = 11
min_disp = 0
max_disp =  48
num_disp = max_disp - min_disp
uniquenessRatio = 5
speckleWindowSize = 0
speckleRange = 2
disp12MaxDiff = 0

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)
disparity = stereo.compute(imgL_bw , imgR_bw )

img = disparity.copy()
plt.imshow(img, 'CMRmap_r')
plt.show()


#  Reading calibration
cam1= np.array([[721.5377, 0, 609.5593],
                [0, 721.5377, 172.540],
                [0, 0, 1] ])
cam2 = np.array([[721.5377, 0, 609.5593 ] ,
		        [0 ,721.5377, 172.8540 ],
                [0, 0, 1 ]])


T= np.array([0.54, 0., 0.])

rev_proj_matrix = np.zeros((4,4))

cv2.stereoRectify(cameraMatrix1 = cam1,cameraMatrix2 = cam2, \
                  distCoeffs1 = 0, distCoeffs2 = 0, \
                  imageSize = imgL.shape[:2], \
                  R = np.identity(3), T =T , \
                  R1 = None, R2 = None, \
                  P1 =  None, P2 =  None, Q = rev_proj_matrix);


points = cv2.reprojectImageTo3D(img, rev_proj_matrix)

#x axisreflection
reflect_x = np.identity(3)
reflect_x[0] *= -1
points = np.matmul(points,reflect_x)

colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
mask = img > img.min() #mindisparity filter
out_points = points[mask]
out_colors = colors[mask]

idx = np.fabs(out_points[:,0]) < 4.5
out_points = out_points[idx]
out_colors = out_colors.reshape(-1, 3)
out_colors = out_colors[idx]

#save out_points,out_colours 
write_ply('cloud.ply', out_points, out_colors)



