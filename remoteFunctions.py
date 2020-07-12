# coding: utf-8

import numpy as np
from matplotlib import pylab as plt


# ================================================================================

def mySize(arrList,dpi = 100):
    ''' Only up to 6 images, 3 images per row.
        Args:
            - arrList:   list of numpy arrays
            - dpi:       piksel for inch
    '''
    dpi = 1/dpi
    h = max([x.shape[0] for x in arrList])
    w = max([x.shape[1] for x in arrList])
    l = len(arrList)
    if l == 1:
        height = h
        width = w
    elif l <= 3:
        height = h
        width = w * l
    elif l > 3 and l < 7:
        height = h * 2
        width = w * 3
    return int(width * dpi), int(height * dpi)

# ================================================================================

def myOtsu(ar):
    ar = ar.copy()
    ar = ar.ravel() # converts 2D arrays to 1 dimensional data vector
    mu = ar.mean()    # total mean value
    N = ar.size
    K = 2
    pixelVal = np.arange(256)
    threshold =[0,0] # collects two values: [sigma, threshold]
    
    # data to the graph as variance changes
    tt = [] # collects threshold values
    sg =[]  # collects variances
   
    for t in pixelVal: 
        cl1 = ar[(ar < t)]
        cl2 = ar[ar >= t]
        
        #if any class has zero elements - skip this threshold
        if cl1.size ==0 or cl2.size == 0:
            continue
        
        N1,N2 = cl1.size, cl2.size
        w1,w2 = N1/N, N2/N  # calculate weight
        mu1,mu2 = cl1.mean(), cl2.mean()
        
        sigma_p = (w1*(mu1-mu)**2 + w2*(mu2-mu)**2)/(K-1)
        sigma_p = np.round(sigma_p,0)
              
        tt.append(t)
        sg.append(int(sigma_p))
        
        # it starts with the value threshold[0] = 0
        if sigma_p > threshold[0]:
            threshold[0]= sigma_p
            threshold[1] = t  
    del ar

    return (tt,sg,threshold)


# ================================================================================

def myplot(images,title=None,size=None,hist=None):
    ''' Only up to 6 images, 3 images per row.
        Args:
            images - list of images (np.arrays)
            titles - list of image titles
            size   - tuple of image size in inch eg. (10,12)
            hist   - says whether to display an image or a histogram
    '''
    imm = images[:]
    l = len(imm)  # number of images
    if l > 3 and l <=6:
        rows, cols = 2, 3  # needs two rows (max 3 images per row)
    elif l <= 3:
        rows, cols = 1, l
        
    if size:
        f = plt.figure(figsize=size)
    else:
        f = plt.figure()
    for i,img in enumerate(images,1): # i - image number from 1
        plt.subplot(rows,cols,i)
        if hist:
            plt.hist(img.ravel(),bins=256)
        else:
            plt.imshow(img, cmap=plt.cm.Greys_r)
            plt.axis('off')
        if title:
            plt.title(title[i-1])
        
    f.tight_layout()


# ================================================================================


def rgbToGray(ar,weights=[0.3,0.59,0.11]):
    gr = ar.copy().astype(np.float64)
    R,G,B = gr[:,:,0], gr[:,:,1], gr[:,:,2]
    weights = weights[:]
    
    R = R * weights[0]
    G = G * weights[1]
    B = B * weights[2]
    
    gr = np.array(R+G+B,dtype=np.uint8)
    return gr




# ================================================================================



def myStretchOld(ar,t):
    ''' Args:
            - ar:   numpy 2D array like image
            - t:    tuple (a, b), a: threshold on the left
                                  b: threshold on the right'''
    ar = ar.copy()
    
    # transformation coefficients: point 1: t, point 2: (245,255)
    a,b = np.polyfit(t,[0,255],1)
    print(f'Transformation coefficients:\n{"a:":>15} {a}, b: {b}\n')
    x = np.unique(ar.ravel())
    y = a*x + b

    new_ar = ar * a + b
    new_ar[ar<0] = 0
    new_ar[ar>255] = 255
    new_ar = np.array(new_ar,dtype=np.uint8)
    
    return new_ar

# ================================================================================

def myNormalize(ar):
    ''' This function is intended to be used as a subfunction of the stretch function.
        Works only on 2D array - single channel.
        Args:
            - ar:  numpy array 2D
    '''
    ar = (ar - ar.min())/(ar.max() - ar.min())
    return ar
    

def subStretch(ar,t,maxVal=1):
    ''' This function is intended to be used as a subfunction of the stretch function.
        Works only on 2D array - single channel.
        Args:
            - ar:     numpy array 2D
            - t:      int 'a' or tuple (a, b) - thresholds
            - maxVal: int, 1 or 255:
                        - 1 for float array
                        - 255 for 8 bits array (np.uinit8)
    '''
    ar = ar.copy()
    g1 = np.percentile(ar.flat,t[0])
    g2 = np.percentile(ar.flat,100 - t[1])
    
    # transformation coefficients: point 1: t, point 2: (245,255)
    a,b = np.polyfit([g1,g2],[0,maxVal],1)

    new_ar = ar * a + b
    new_ar = np.round(new_ar,6)
    new_ar[new_ar<0] = 0
    new_ar[new_ar>maxVal] = maxVal
    
    return new_ar
    
    
    
def myStretch(ar,t=1,model='rcb'):
    ''' Args:
            - ar:   numpy 2D or 3D array like image
            - t:    int 'a' or tuple (a, b) - thresholds
                      - if a: symetric threshold
                      - if (a,b): a - left, b - right
            -model: int,
                      - 1: (rows,cols,n_bands)
                      - 2: (n_bands,rows,cols)
                      '''
    dataType = ar.dtype.name
    # check data type and set maxVal
    #if 'uint8' == dataType:
    #    maxVal = 255
    #else:
     #   maxVal = 1
 
    ar = ar.copy().astype(np.float32)

    # convert int tu tuple of int for right and left thresholds
    if isinstance(t,int):
        t = (t,t)
    
    # split function for:
    # ---- single band images ....................
    if len(ar.shape) == 2:
        if dataType in ['uint16','uint8']:
            ar = myNormalize(ar)
        # warning!!! array with float but > 1 is not walid yet
        new_ar = subStretch(ar,t)

    #---- multibands images ......................   
    else:
        if model=='brc': # brc to rcb
            ar = np.transpose(ar,(1,2,0))

        rows,cols,bands = ar.shape
        #print(rows,cols,bands,'\n!!!!!')
        new_ar = np.zeros([rows,cols,bands],dtype=np.float32)
        
        #rows,cols,bands = new_ar.shape
        #print(rows,cols,bands,'\n!!!!!')
        
        for i in range(bands):
            band = ar[:,:,i].copy()
            if dataType in ['uint16','uint8']:
                band = myNormalize(band)
            new_ar[:,:,i] = subStretch(band,t)

    del ar, band
    
    # set data type for 8bits images (from 0 to 255)
    #if dataType == 'uint8':
        #new_ar = np.array(new_ar,dtype=np.uint8)
        
    return new_ar

# ================================================================================

def myEqual(ar,c=0):
    ''' Args:
            - ar:   numpy 2D array like image'''
    ar = ar.copy()
    w,h = mySize([ar])
    new_ar = ar.copy()
    freq,bins = np.histogram(ar.ravel(),bins=256,range=(0,255),density=False)
    
    # D: calculate CDF
    D = np.cumsum(freq)/freq.sum()
    
    # if c is True (1) - plot CDF
    if c:
        plt.plot(bins[1:],D);
        plt.title('Empirical cumulative distributive function (CDF)')
    
    D_min = D[D>0].min() # get smallest value of 𝐷, greater than zero
    nk = 256         # number of possible image values

    # Look Up Table (LUT)
    LUT = ((D - D_min)/(1-D_min))*(nk-1)
    LUT[LUT < 0] = 0
    LUT = np.round(LUT,0).astype(np.uint8)
    
    # set new pixel value with new array
    for i in range(LUT.size):
        new_ar[ar==i] = LUT[i]
        
    return new_ar



# ================================================================================


def myKmeans(ar,k,n=10,atype='rcb'):
    ''' Args:
            - ar:     np.array 2D or 3D
            - k:      noumber of classes / centriods
            - atype:  type of shape:
                        - 'rcb': row, cols, bands
                        - 'brc': bands,rows,cols
    '''
    ar = ar.copy()
    
    dataType = ar.dtype.name
    
    with np.errstate(divide='ignore', invalid='ignore'):
    
        # check data type and set maxVal
        if dataType in ['uint8','uint16']:
            ar = myNormalize1(ar,atype)
            print(f'dataType: {dataType} --> normalized')
            
        # conwert array to data - see:
        #       'Preparation of data for classification'       
        if len(ar.shape)>2:
            if atype == 'brc':
                bands,rows,cols = ar.shape
                data = ar.T.reshape(-1,bands)
                print(f'brc-shape: {ar.shape}')
            else:
                rows,cols,bands = ar.shape
                data = ar.reshape(-1,bands)
                print(f'rcb-shape: {ar.shape}')
        else:
            rows,cols = ar.shape
            bands = 1
            data = ar.reshape(-1,1)
            print(f'one band shape: {ar.shape}')
            
        print(f'\nData for calculation!\ndata.shape: {data.shape}\n\tdata[:4,:]:\n{data[:4,:]}\n')
        # initialize centroids:
        # k - noumber of centriods,  bands - dimension of coordinates
        #cent = np.random.rand(k,bands)
        #cent = np.random.normal(data.mean(),data.std(),(k,bands))
        cent = np.random.randint(0,data.shape[0],k)
        cent = data[cent,:]
        print(f'initial cetroids - shape:{cent.shape}\n\n{cent}')
        centOld = np.zeros_like(cent)
        err = (centOld - cent)**2
    
        it = 0
        #while err.sum() != 0:
        for _ in range(n):
            it +=1
            labels = nearCentroid(data,cent)
            print(f'np.unique(labels): {np.unique(labels)}, {labels.shape}')
            centOld = cent.copy()
            for i in range(k):
                print(f'i: {i}, labels==i: {labels[labels==i].size}')
                
                
            cent = [data[labels==i].mean(axis=0) for i in range(k)]
            cent = np.array(cent)
            
            
            #print(cent)
            err = ((centOld - cent)**2).sum(axis=0)
            err = np.round(err,5)
            print(f'Iteration no: {it}, error: {err}\n')
        
    classIm = np.array(labels).reshape(rows,cols)
    classIm = (classIm/classIm.max())*255
    classIm = np.array(classIm,dtype=np.uint8)
    #classIm = np.array(labels,dtype=np.uint8).reshape(rows,cols)
    return classIm




def myNormalize1(ar,atype='rcb'):
    ar = ar.copy()
    if len(ar.shape) > 2:
        arn = np.zeros_like(ar,dtype=np.float32)
        if atype == 'brc':
            bands = ar.shape[0]
            for i in range(bands):
                bb = ar[i,:,:]
                arn[i,:,:] = (bb - bb.min())/(bb.max() - bb.min())
            
        else:
            bands = ar.shape[-1]
            for i in range(bands):
                bb = ar[:,:,i]
                arn[:,:,i] = (bb - bb.min())/(bb.max() - bb.min())
    return arn



def nearCentroid(ar,centroid):
    ar = ar.copy()
    c = centroid.copy()
    diff = (ar - c[:,np.newaxis])**2
    distance = (np.sum(diff,axis=2))**0.5
    minIdx = np.argmin(distance,axis=0)
    return minIdx


