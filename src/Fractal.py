# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com July 2010

# Based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

import sys
import numpy as nm

import Tkinter as tk
import Image          # PIL
import ImageTk        # PIL
import random
import cv2

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

# set width and height of window, more pixels take longer to calculate
scale = 3
w = 350*scale
h = 300*scale
zoomFactor = 0.5
frames = 60

imgs = []

from pycuda.elementwise import ElementwiseKernel
complex_gpu = ElementwiseKernel(
        "pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter",
            "for (int n=0; n < maxiter; n++) {z[i] = (z[i]*z[i])+q[i]; if (abs(z[i]) > 2.0f) {iteration[i]=n; z[i] = pycuda::complex<float>(); q[i] = pycuda::complex<float>();};}",
        "complex5",
        preamble="#include <pycuda-complex.hpp>",)

def calculate_z_gpu(q, maxiter, z):
    output = nm.resize(nm.array(0,), q.shape)
    q_gpu = gpuarray.to_gpu(q.astype(nm.complex64))
    z_gpu = gpuarray.to_gpu(z.astype(nm.complex64))
    iterations_gpu = gpuarray.to_gpu(output)
    # the for loop and complex calculations are all done on the GPU
    # we bring the iterations_gpu array back to determine pixel colours later
    complex_gpu(z_gpu, q_gpu, iterations_gpu, maxiter)

    iterations = iterations_gpu.get()
    return iterations


def calculate_z_numpy_gpu(q, maxiter, z):
    """Calculate z using numpy on the GPU via gpuarray"""
    outputg = gpuarray.to_gpu(nm.resize(nm.array(0,), q.shape).astype(nm.int32))
    zg = gpuarray.to_gpu(z.astype(nm.complex64))
    qg = gpuarray.to_gpu(q.astype(nm.complex64))
    # 2.0 as an array
    twosg = gpuarray.to_gpu(nm.array([2.0]*zg.size).astype(nm.float32))
    # 0+0j as an array
    cmplx0sg = gpuarray.to_gpu(nm.array([0+0j]*zg.size).astype(nm.complex64))
    # for abs_zg > twosg result
    comparison_result = gpuarray.to_gpu(nm.array([False]*zg.size).astype(nm.bool))
    # we'll add 1 to iterg after each iteration
    iterg = gpuarray.to_gpu(nm.array([0]*zg.size).astype(nm.int32))

    for iter in range(maxiter):
        zg = zg*zg + qg

        # abs returns a complex (rather than a float) from the complex
        # input where the real component is the absolute value (which
        # looks like a bug) so I take the .real after abs()
        abs_zg = abs(zg).real

        comparison_result = abs_zg > twosg
        qg = gpuarray.if_positive(comparison_result, cmplx0sg, qg)
        zg = gpuarray.if_positive(comparison_result, cmplx0sg, zg)
        outputg = gpuarray.if_positive(comparison_result, iterg, outputg)
        iterg = iterg + 1
    output = outputg.get()
    return output

if len(sys.argv) > 1:
    if sys.argv[1] not in ['gpu', 'gpuarray', 'numpy', 'python']:
        show_instructions = True





##if sys.argv[1] == 'python':
#    import psyco
#    psyco.full()

# Using a WinXP Intel Core2 Duo 2.66GHz CPU (1 CPU used)
# with a 9800GT GPU I get the following timings (smaller is better).
# With 200x200 problem with max iterations set at 300:
# calculate_z_gpu: 0.03s
# calculate_z_serial: 8.7s
# calculate_z_numpy: 0.3s
#
# Using WinXP Intel 2.9GHz CPU (1 CPU used)
# with a GTX 480 GPU I get the following using 1000x1000 plot with 1000 max iterations:
# gpu: 0.07s
# gpuarray: 3.4s
# numpy: 43.4s
# python (serial): 1605.6s

class Mandelbrot(object):
    def __init__(self):
        self.originX = -2.5
        self.originY = 1.5
        self.xLength = 3.5
        self.yLength = 3
        # create window
        self.root = tk.Tk()
        #self.root.tk.call('tk', 'scaling', 2.0)
        self.root.title("Mandelbrot Set")
        self.create_image(self.originX, self.originX+self.xLength, self.originY-self.yLength, self.originY)
        self.create_label()
        self.root.bind('<Button-1>', self.buttonPressed)
        # start event loop
        self.root.mainloop()


    def draw(self, x1, x2, y1, y2, maxiter=300):
        #print("x1: {}, x2: {}, y1: {}, y2: {}, xLen: {}, yLen: {}, w: {}, h: {}".format(x1,x2,y1,y2, self.xLength, self.yLength, w, h))

        # draw the Mandelbrot set, from numpy example
        xx = nm.arange(x1, x2,(x2-x1)/w)
        yy = nm.arange(y1, y2, (y2-y1)/h) * 1j
        # force yy, q and z to use 32 bit floats rather than
        # the default 64 doubles for nm.complex for consistency with CUDA
        yy = yy.astype(nm.complex64)
        q = nm.ravel(xx+yy[:, nm.newaxis]).astype(nm.complex64)
        z = nm.zeros(q.shape, nm.complex64)

        start_main = drv.Event()
        end_main  = drv.Event()
        start_main.record()
        output = calculate_z_gpu(q, maxiter, z)

        end_main.record()
        end_main.synchronize()
        secs = start_main.time_till(end_main)*1e-3
        print "Main took", secs

        output = (output + (256*output) + (256**2)*output) * 8
        # convert output to a string
        self.mandel = output.tostring()

    def create_image(self, x1, x2, y1, y2):

        self.im = Image.new("RGB", (w, h))
        # you can experiment with these x and y ranges
        self.draw(x1, x2, y1, y2, 1000)

        self.im.frombytes(self.mandel, "raw", "RGBX", 0, -1)

    def create_label(self):
        self.image = ImageTk.PhotoImage(self.im)
        self.label = tk.Label(self.root, image=self.image)
        self.label.pack()

    def calcIncrement(self, prev, new, nbr):
        return (new-prev)/nbr


    def buttonPressed(self, event):
        print("Pressed at [ " + str(event.x) + ", " + str(event.y) + " ]")

        global w
        global h

        xRatio = float(event.x) / float(w)
        yRatio = float(event.y) / float(h)

        deltaX = xRatio * float(self.xLength)
        deltaY = yRatio * float(self.yLength)

        xClickPt = self.originX + deltaX
        yClickPt = self.originY - deltaY

        newOriginX = xClickPt - (self.xLength * 0.5 * zoomFactor)
        newOriginY = yClickPt + (self.yLength * 0.5 * zoomFactor)
        newxLen = self.xLength * zoomFactor
        newyLen = self.yLength * zoomFactor

        deltaOx = (newOriginX - self.originX)/frames
        deltaOy = (newOriginY-self.originY)/frames
        deltaXLen = (newxLen -self.xLength)/frames
        deltaYLen = (newyLen-self.yLength)/frames

        for i in range(frames):
            tmpOx = (deltaOx * i) + self.originX
            tmpOy = (deltaOy * i) + self.originY
            tmpDeltaXLen = (deltaXLen * i) + self.xLength
            tmpDeltaYLen = (deltaYLen * i) + self.yLength
            self.create_image(tmpOx, tmpOx+tmpDeltaXLen, tmpOy-tmpDeltaYLen, tmpOy)
            imgs.append(self.im)

        self.originX = xClickPt - (self.xLength * 0.5 * zoomFactor)
        self.originY = yClickPt + (self.yLength * 0.5 * zoomFactor)

        self.xLength *= zoomFactor
        self.yLength *= zoomFactor

        self.create_image(self.originX, self.originX+self.xLength, self.originY-self.yLength, self.originY)
        self.label.destroy()
        self.create_label()

    def renderVideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))

        for img in imgs:
            img = img.convert('RGB')
            img = nm.array(img)
            video.write(img)

        cv2.destroyAllWindows()
        video.release()

# test the class
if __name__ == '__main__':
    mandelbrot = Mandelbrot()
    mandelbrot.renderVideo()



