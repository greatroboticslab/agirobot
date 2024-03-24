from PIL import ImageGrab, ImageEnhance, ImageFilter
import win32gui

#width = 256
#height = 210

width = 224
height = 224

contrast = 5.0

toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, toplist)

rainworld = [(hwnd, title) for hwnd, title in winlist if 'rain world' in title.lower()]
print(rainworld)
rainworld = rainworld[0]
hwnd = rainworld[0]

def CaptureFrame():
    
    win32gui.SetForegroundWindow(hwnd)
    bbox = win32gui.GetWindowRect(hwnd)
    img = ImageGrab.grab(bbox)
    
    img = img.resize((width, height))
    
    #enhancer = ImageEnhance.Contrast(img)
    #img = enhancer.enhance(contrast)
    img = img.filter(ImageFilter.FIND_EDGES)
    
    return img
    #img.show()
    
def CapturePosFrame():
    
    img = CaptureFrame()
    
    rect = win32gui.GetWindowRect(hwnd)
    _x = rect[0]
    _y = rect[1]
    
    (x,y) = win32gui.GetCursorPos()
    
    x -= _x
    y -= _y
    
    return img, (x,y)