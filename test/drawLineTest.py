from PIL import Image,ImageDraw

img= Image.open("C:\\Users\\14542\\Desktop\\test.jpg")
img_d = ImageDraw.Draw(img)
img_d.line((20, 20, 150, 150), 'cyan')
img.save("C:\\Users\\14542\\Desktop\\test2.jpg")