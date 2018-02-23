import matplotlib.pyplot as mp
import numpy as np


class image(np.ndarray):
    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)
	



		
#############################################################################




def setPixel(image, x,  y, c):
	image[x][y] = c
	return image


def imread(filename):	
	image = mp.imread(filename)
	if image.dtype == 'float32':
		image.dtype = np.uint8
		image * 255
	if nchannels(image) > 3: #trata caso em que imagens tem mais que 3 canais
		return toRBG(image)
	return image
	
def imshow(image):
	if(nchannels(image) == 1):
		mp.imshow(image, cmap = 'gray', interpolation = 'nearest')
	else:
		mp.imshow(image,cmap = 'gray')
	
	mp.show()

def nchannels(image):
	dim = image.shape
	last = len(dim) - 1
	return dim[last]

def size(image):
	dim = image.shape
	return np.array(dim)
	

def rgb2gray(image):
	tam = size(image)
	
	x = tam[0]
	y = tam[1]
	print(x)
	print(y)
	result = np.ndarray(shape=[x,y])
	for row in range(0,x):
		for col in range(0,y):
			mul = image[row][col] #[[0.299],[0.587],[0.1114],[0]]
			
			mul[0] *= 0.299
			mul[1] *= 0.587
			mul[2] *= 0.1114
			
			
			result[row][col] = sum(mul)
	return result

def imreadgray(name):
	img = imread(name) # read the image
	if(nchannels(img) > 1):
		return rgb2gray(img) 
	return img

def toRBG(image):
	return image[:,:,0:3]

def thresh(image, limiar):
	"""dims = size(image)
	x = dims[0]
	y = dims[1]
	#print(image)
	img = np.ndarray(shape=[x,y,3])
	
	for linha in range(0,x):
		for coluna in range(0,y):
			l = [255 if y >= limiar else 0 for y in image[linha][coluna]]
			#lim[0] = 255 if lim[0] >= limiar else 0
			#lim[1] = 255 if lim[1] >= limiar else 0
			#lim[2] = 255 if lim[2] >= limiar else 0
			#img[linha][coluna] = lim
			print(l)
			img[linha][coluna] = l
			
	return img"""
	return ((image >= limiar) * L).astype(np.uint8)


def negative(image):
	"""dims = size(image)
	x = dims[0]
	y = dims[1]
	#print(image)
	img = np.ndarray(shape=[x,y,3])
	
	for linha in range(0,x):
		for coluna in range(0,y):
			img[linha][coluna] = [abs(pixel - 255) for pixel in image[linha][coluna]]
	

	return img"""
	L = 255
	return (L - image).astype(np.uint8)





def histFromArray(array):
	unique, counts = np.unique(array, return_counts=True)
	return counts #dict(zip(unique,counts))

def hist(image):
	histogram = [0 for x in range(0, 256)] # it goes from 0...255
	matriz = []
	dims = size(image)
	x = dims[0]
	y = dims[1]
	channels = 1
	if(len(dims) == 2):
		#gray image
		channels = 1
	else:
		#rbg image
		channels = 3
	
	if(channels == 1):
		"""for linha in range(0,x):
			for coluna in range(0,y):
				pixel_value = image[linha][coluna]
				countValue = histogram[pixel_value] + 1 ##increment
				histogram.insert(pixel_value,countValue) """
		unique, counts = np.unique(image, return_counts=True)
		return counts

		
	elif(channels > 1):
		r = image[:,:,0:1]
		g = image[:,:,1:2]
		b = image[:,:,2:3]

		return [histFromArray(r),histFromArray(g),histFromArray(b)]
		

def plotHist(data):
	if(len(data) == 1):
		mp.hist(data,50,density=True, facecolor='g', alpha=0.75)
	else:
		mp.hist(data[0],50,density=True, facecolor='r', alpha=0.75)
		mp.hist(data[1],50,density=True, facecolor='g', alpha=0.75)
		mp.hist(data[2],50,density=True, facecolor='b', alpha=0.75)
	mp.show()

def contrast(img, r, m):
    newImg = img.copy()
    if (nchannels(img) == 1): 
        for x in range(0, len(newImg)): 
            for y in range(0, len(newImg[0])):
                tmp = r*(img[x][y]-m)+m
                if (tmp >= 255):
                    newImg[x][y]  = 255
                elif (tmp <= 0):
                    newImg[x][y] = 0
    else: 
        for x in range(0, len(newImg)): 
            for y in range(0, len(newImg[0])):
                tmp = r*(img[x][y][0]-m)+m
                if (tmp >= 255):
                    newImg[x][y][0]  = 255
                elif (tmp <= 0):
                    newImg[x][y][0] = 0
                tmp = r*(img[x][y][1]-m)+m
                if (tmp >= 255):
                    newImg[x][y][1]  = 255
                elif (tmp <= 0):
                    newImg[x][y][1] = 0
                    tmp= r*(img[x][y][2]-m)+m
                if (tmp  >= 255):
                    newImg[x][y][2]  = 255
                elif (tmp <= 0):
                    newImg[x][y][2] = 0
    return newImg


def maskblur():
    return [[0.0625,0.1250,0.0625],[0.125,0.2500,0.1250],[0.0625,0.1250,0.0625]]
    
def blur(img):
    return convolve(img,maskblur())

def seSquare3():
    return [[1,1, 1], [1, 1, 1], [1, 1, 1]]

def seCross3():
    return [[0, 1,0], [1, 1, 1], [0, 1, 0]]





'''
im = imread('forest.png')
image2 = thresh(im, 25)
#im2 = rgb2gray(im)
print(image2.shape)
imshow(image2)
#ch = nchannels(im)
#dims = size(im)

#print("Canais " + str(ch))

#print("dimensoes " + str(dims))	

'''