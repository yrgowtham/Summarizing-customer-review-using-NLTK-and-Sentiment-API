from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import wordnet as wn
import  testapi
from textblob import TextBlob
import nltk
import numpy as np
example_txt= """
This is my first DSLR camera. I bought for a photography course. If you are thinking of buying the kit with two lenses, buy an 18-250mm or a 18-300mm lens instead. Sony, Tamron, and Sigma make pretty decent lenses for under $550.
I've had my A350 for over a month.  I shot about 400 photos with it over the last weekend.  I've hesitated to use it, due to what I presumed was bad design.  It turns out that my camera has a misaligned viewfinder.  I was having to aim lower, guessing what was the proper place to point at, and then take my shot.  Sometimes I got what I was trying to photograph.  Sometimes the image was cut off.  Looks like this camera is going back for repairs.  After that, who knows, it might be a great camera.
I was always a Nikon fan but when I saw the Alpha 350 and read about its reatures I was truly impressed. For a lot less than a grand it seemed to have features that Nikon only had in $2500 to $3000 models and sometimes not even in them. I really loved the live view and the tilting view screen. Nikon didn't have that at the time and I'm not sure they even had live view. I have a 10 year old Sony touch screen handy cam that to me still seems "state of the art" other than not taking DVDs. I'm not sure that's so bad considering how easily DVDs scratch. The hanycam is really nice because you can turn the screen so you can see yourself as you are recording. I loved this because I was hiking the AT at the time and could do commentaries of myself. You can't turn the viewfinder of the Alpha 350 so you can take pictures of yourself but you can turn it down so you can hold it above your head and take pictures above the crowd. You can move it so it is looking up so you can hold it near ground level and that give you a "dogs eye" view. That's one of the few things that sold me about the camera.Steady shop is built into the camera body so the lenses are less expensive, you can use Minalta and some other lenses. The menus are very easy to use and all the bottons are nicely placed. I could go on and on but there are plenty of other reviews on this camera. I also got it at a great deal from Amazon.I only wish I had waited 3 years because I am now in love with the Alpha 55. Only kidding, I couldn't have waited. But the Alpha 55 seems, for a non professional camera to be way ahead of what Nikon has in its price range.I think when Nikon was actually made in Japan it was the best. But now this is not the case and they  make it were ever they can get the best price. Look at the amount of reviews they have for their cameras. A camera might have 3000 reviews. Can a company pump that many camera out and really maintain the quality they use to be famous for. Obviously there are still a lot of people who are brand oriented and aren't really looking at the quality and features. Kind of like republicans. The only good republican was the first one and that was Lincoln. But people are still brand loyal regarless of the product. I bought my first Nikon when I was in the Marines back in 1963 or so. Back then it was strictly the Nikon F and I truly believe it was the greatest work of art I ever owned. But I think times have changed.I better stop boring you people. I truly love the Alpha 350. You probably can't find them unless you buy them used, but most people care for there cameras with kid gloves so if you bought a used one it would probably be like new. This camera is a work of art, it's sitting on my coffee table and I still admire its quality. Needless to say I recommend it and most other sonys.
I purchased this camera to provide photo documentation of projects I perfomed through my shop (antique restorations).  The camera itself performed admirabley until Christmas morning.  I have treated this camera with all do respect, it has never been dropped, thrown about or misused in any way.  The memory card interface has failed - the camera will not recognize the card (the card has been verified to be in good working order).  So after only 16 months of light duty and proper care it is completely useless.  Sony wants me to pay $288 plus freight to repair a failure which is clearly a manufacturing or product design defect.  I'm sorry but for what I paid for this camera I should not be experiencing this problem never mind having to pay to get it resolved.ps:  My last SLR cost half as much and lasted over 20 years without a single failure.  I will NEVER buy another Sony product!!
First let me say I did not purchase this from Amazon (a first for me since I purchase everything from here). I did find a better deal at Circuit City BUT  I would NOT recommend buying from there again. Very lousy customer service and little sales knowledge of product. Luckily I did some research before going there.Now about the camera. I LOVED all the features and the ease of operation. I loved the focusing and the best feature I found to be the LCD screen that tilts allowing you to view images while holding camera above you or taking low shots.What I did not like was the lack of picture quality. I found the pictures to be VERY blurry and not in focus. I tried manual focusing and auto focusing. I tried to make sure the camera had no movement when shooting. I tried everything. Out of the 200 or so shots I took with it I can say that only 10 or 15 came out acceptable. I love close up photography of flowers & insects, I couldn't do it with this camera.I really wanted this camera but the image just made it unacceptable.I bought this camera because I have a 1970 Minolta SRT 101 35mm camera with wide angle and telephoto lens. I understood that I could use my lens from Minolta with this Sony A350. I found out this would not work. I found that I like the camera and the lens I bought. I am still learning how to use it.
Writing this to help a new buyer to make a decision :-)The Goods ....1. 14.2 Megapixel ....the best in its class and price range2. Live view LCD....a must have feature for thos graduating from point and shoot digital cameras3. Anti shake control (image stabilization) in camera body .... Can use any lens compared to canon rebel and Nikon where the image stabilization is in the lens (which is expensive)4. Good form factor ...fits nice in hands and Stylish5. LCD tilts out ....very nice feature which opens up a lot of new "camera angles" for the picture.6. Very fast autofocusing ....and produce really great pictures7. Options to shoot RAW, Jpeg & RAW+Jpeg8. Controls are very easy to use.....9. Even beginners can take amazing pictures with this camera....dont get fooled by those pictures in various websites....( i had this doubt before buying and later found that the canon and nikon pictures given in net are mostly taken by hardcore proffessionals...where as sony pictures are from beginners )10. Battery last long...it is of least worry11. Nice fitting of parts .....and no squeaking sounds....except a little inside the grip under the trigger.The Bads....1. Slowest continous shooting in its class !! if you like your camera to take 3-4 pictures in a second go for cannon or nikon. This shoots continous pictures almost like a point and shoot.2.Auto Exposure bracketing step is 0.7 which is way too Low in comparison to canon rebel and nikons....if you are into HDR this is not the camera3.Sony lenses are very expensive and have limited choices....well i am yet to try others like tamaron or minolta .....carlZeis is too expensive !4. the strap supplied with the camera is lousy5. Body made in Japan...kit lens made in china ....and it shows the quality difference.6. Flash timing is few micro seconds slow...then what ? some "subjects" who have higher reflex action will blink their eyes and they look "sleepy" in the pictures (I always blink and my wife dont so i look sleepy in all pics)7. this camera and all SLRs use Compact flash card!! No SD,No M2, No produo, no micro SD......I realised this only after getting the camera and i didnt find this info in anyother review so i am writing it.so better order a 4gb card ( yes you need that) with 30mb/s speed when you buy this camera.Tips1.dont forget to buy a LCD protecting cover along with this camera ...LCD is tiltable and generic covers wont fit.2. dont forget to buy a filter either UV or polarizer ....this will help you to keep finger prints out of your lens...and it is lot cheaper to change a broken filter than to change a Lens!3. Dont get fooled by the high iso noise talk in net.....all cameras produce noise when in high ISO ....4. This camera can take great pictures with stunning quality but5. Remember that Buying an SLR will not make you a Photographer...it will only make you an SLR owner !!!Happy buying .......Enjoy this Camera !!!it smells quality !By the way i uses 18-70 kitlens ....and so far happy ...and enjoying the ride ;-)UPDATE after 2years :  Excellent Camera Love it still !!have taken many stunning shots with this and few of my pictures found its way to magazine cover pages and one picture won a competition too.had an issue of a dust spec showing up in all pics but the sony service corrected it for me, i paid only shippingLens are still an issue...limited choices and expensive...Live view is great...and now every DSLR maker follows this example set by sony in this cameraone dead pixel in the display ...but it is OK.
Wonderful camera, but the lens not so good. I'm looking for a new lens to replace the original one.
having enjoyed my vivitar point 'n shoot until it broke, I was searchng for a new camera. almost bought a Nikon, until I saw the A350, 14.2 megapixels WOW beats cameras costing more. I love macro photography. The standard lens it comes with I don't think I've used it more than twice. I gota TAMRON 70/300MM 1:2 macro lens. In just ten weeks I have taken over 5000 pictures. Some of the best pictures I've ever taken. My friends say so too. Sony's website is great for tutorials. I work with color and light all day, so I think I have at least a little leg up.  This camera takes a great leap up.  Read the manual well. I read it, went to take pictures and when something turned out wrong I'd go back to the manual to figure why. The clarity, sharpness of the pictures are incredible! Cons? none that I can think of !
This camera is best for professional photographers as well as learners. The picture quality is awesome. I love having this camera.
The best thing about this camera is that it is super easy to use in the normal auto mode. It also has a lot of other special settings for custom shots for maximum creativity. The Super Steady Shot image stabilization reduces blur. The screen is tiltable for high and low angle shots. You can use Live Preview on a large w.7 LCD screen.  The camera allows you continous shooting at 2 fps (frames per second) while you are viewing your subject in the view finder. You can also check out the high resolution thumbnails. Auto pop-up flash has delayed flash to eliminate red eye issues. 14.2 MP for ultra high resolution.  I really don't have much to say about the lenses because I swapped mine out for some prefered ones once I got the camera.
I loved my purchase of a Sony A-350 camera, however, my colleague, Roxanee Grunig of Gualala, CA  wanted the same camera and was swayed by a New York City Amazon.com camera owner and was told it was dusty. He convinced her, for the same money to purchase a camera A-390 of lesser value . She agreed to the purchase but I think she got horn swaggered.My A-350 cost $1,600 new and the A-390 cost $525. new.  I paid $331.00 for my A-350 used and Roxanne got her used A-390 with accessories for $350.She has not received her camera yet, but the sale still bothers me. Hence, this message.My A-350 was also very dusty with red hairs coming out of the mirror area when I got it, but I blew it out and it was no problem. I did inform the sender to be more careful.
To start off, I don't really write reviews on products, and I am new to digital photography, so you can really take this review with a grain of salt... However, I just thought I would share why I gave this product a 5 star rating.The five star rating simply put is because I love this camera. As stated earlier I am new to digital SLR photography so I won't be nearly as insightful as the Nikon/Canon buffs.Why get the A350? 14.2 megapixels. Live view. Ease of use. Super Steady Shot built into the camera.The high megapixels enable you to take high quality photos that will look great when blown up. Does this mean that this camera will take better pictures than a Nikon D300 because the megapixel count is higher... No, not at all that was a common misconception I learned. But knowing that I wanted to take a lot of pictures and make large digital prints to make my office look less lame, I figured I should go a step higher to get the blown-up image quality that I want.The live view is great for so many reasons. Right on the display, it tells you all your settings from battery life, flash, aperture, etc... all while giving a great idea what your image will look like. Keep in mind, what you see on the live view isn't 100% what you'll see when you load the photos on computer or photoshop.Ease of use: Grab the camera, go over the manual, play with the settings, take pictures. Coming from someone who knew nothing about cameras until about 2 weeks ago, the learning curve was easy with this camera. Right out of the box, you can set everything to auto and take great pictures just like a small digital camera. But for more advanced photos you simply learn by trial and error and the live view helps you see what you might need to adjust when taking photos. So even if you've never touched an SLR camera, you begin to learn what ISO, f/5.6, and all the other settings are and how to use them just by taking good pictures... or bad ones!Super Steady Shot: Clear images, built into the camera, not the lens like other manufacturers. Down the line it can help you save some money because you can buy less expensive lenses because you don't need to pay the extra money to get the image stabilization in the lens like Canon or Nikon.So why get this camera vs. other brands? First off, this camera is great for anyone wanting to get into digital SLR's. If you have old Minolta A-mount lenses you can use them with this camera. There are websites that give compatibility charts with some of the old lenses that you have.If you have a bunch of DSLR Nikon and Canon gear, well that sucks for Sony because you won't be buying an A350 anytime soon. If you did, then you're either dumb or have a lot of free time and money to blow. However, if you're thinking about getting a camera, Sony makes a great product and offers tons of features in the camera at a low cost. Down the road, Sony will launch more professional and higher priced cameras so if you're starting out on a DSLR journey, you can begin it with Sony and start collecting lenses, tripods, filters and all the other happy stuff that goes into the expensive hobby of digital photography.Side note: I think this camera was $100 cheaper when I bought it 2 weeks ago. I got the Sony A350X kit (The X is the 2 lens kit, the A350K kit is a 1 lens kit) so I don't know what the deal is there. But all in all, it's a great buy.Oh yeah, if you're trying to decide whether or not to buy the A300 or A350, in case you didn't know the only difference, literally is the megapixels. So unless you are going to blow up your images, go save yourself $200 bucks and get the A300. If you're going to make large prints like I am, then go with the A350.
I chose the Sony a350 in a very close call over the Canon Rebel XSi. A friend bought the Canon, and we are both happy, so it comes down to preferences. Here is my personal experience with the Sony, with known differences compared with the Canon noted.Please note that I am a hobbyist, rather than a professional.Pros: takes beautiful, clear, and sharp pictures even with the kit lenses. If it were on picture quality alone, the a350 would get a definite 5. This is the most important point of the review. The picture quality is simply awesome.The Sony has excellent low light performance because of the 3200 iso speed (the Canon only goes to 1600, which I have found makes a definite difference in these situations). There is some noise, but it is acceptable to me.The tiltable LCD screen is a nice touch.Battery life is good, and the controls are easy to use after a short learning curve.While it has the highest resolution (14.2MP) of any camera in its class, I'm not sure how much of an advantage this is, since I usually don't need to shoot at that resolution, plus I've seen some reviews where the detail on the Canon in certain settings is actually better.CONS: The biggest disappointment with this camera by far is the built-in flash which utilizes a pre-flash that causes the dreaded sleepy eye effect in most pictures taken in dark settings. I have young children, and they tend to be affected greatly by this feature. I can't seem to find a way to defeat the pre-flash. The problem is much less pronounced when it is used for fill-in in the daytime. This makes the 3200 iso much more important because I now elect to shoot at high speed and no flash. I get much better results this way. Apparently, the Canon does not have this problem quite as much, although it has pre-flash as well.The continuous burst shooting speed is rated at somewhere between 2-2.5 shots per second, slower than the Canon (more in the 3-3.5 range). So far, it hasn't been that big of a deal for me, but I don't usually shoot sports or similar activities.The Sony is a bit heavier than the Canon, but it would be hard to be cognizant of the difference unless you owned both cameras.BOTTOM LINE: I am mostly satisfied with this camera, except for the annoying flash qualities. If you shoot sports a lot, go with the Canon. If you like the ability to shoot in low light settings without flash and shooting with the camera at different angles, go with the Sony.
I don't have several hours to do a proper review on this excellent camera, but I can quickly share the high points.  It has great "human factor" engineering, meaning it fits your hands like a glove and all the controls are easy to access.  The camera/lens combo is very light and easy to handle.  If you have any Minolta lenses, they fit.  The ability to view the picture on the large, bright LCD instead of through the viewfinder is a definite plus in many situations.  14+ megapixels ... really, how many more do you need?Here's the bottom line ... I worked for Kodak, I've been a serious amateur photographer for decades, I've shot thousands of pictures (hundreds so far with this camera) and it leaves nothing that I can think of to be desired.  If you want to spend more bucks on a Nikon or Canon, have fun.  But for my money, this little beauty is a clear winner.
Watch out cannon,pentax and all others. this camera is feature rich and very easy to use with extremely good picture quality.You can not get a better camera in this price range and it even competes with pro cameras.Camera is very easy to use for even a beginner with most all the feachers of $1500.00+ pro cameras.very nice blend of a point and shoot with dslr.This is a very good camera for anyone wanting to improve over there point and shoot picture taking without having to be a pro.Its well thought out and very easy to use at a price thats not much of a stretch from point and shoots.I cant see anyone being disappointed with this camera from armatures to pros alike. GOOD JOB SONY
Not much can be said that hasn't been said.  One Hell of a nice camera.  Took out of box and began shooting excellent photos right off the bat.  I did have to read the instructions to see what a couple of the control were and how to get to a couple of things.  Excellent feel of the camera, Excellent location of control buttons.  Love the anti-shaking, and live view.  Owner of approximately 4 different sonys and have loved each, but the 350 is definitely the top dog.  Have use the Nikon, and canon, they don't hold a candle to this camera.  The company that sent the camera was also excellent, no problems, camera was waiting for me when I got home, no dented boxes.One recommendation is to buy the added lens.
This Alpha  350K, did function very well, it was very handy and convenient to use the tilting screen,the responses are just fine, and I got used to the slow response when using the tilting screen.I was able to take shots over my head at times or over crowds! Also was able to take unusual angles!Also noted that the battery "metering"indicator show the drain/remaining charge, that it uses morebattery with the tilting screen on.I carry an extra battery anyway. We took excellent images with this camera!
I have used this for almost two years and I has been all over the world. Takes great pictures and has held up will being in boats and rough terrian from Japan, Philippines and Dubai. great camera.
A350 was my first DSLR... I only buy it because of it's Quick AF Live View feature: Very Good, like a compact camera.I just don't use the viewfinder... I Don't like to put my eye directly on the camera !!!I bought a 100mm Macro Lens... but the photo quality is not what I've expected... maybe I need some time to improve my shots!The LCD is Bad in Outdoors... you just can't see anything - In comparison with Sony DSC-H9's LCD.No Trimming option on the menu is Bad!
fue un placer esta compra , proceso rapido de mi pago y entrega en la fecha establecida. amazon sigue siendo mi sitio preferido de compras.
I bought the Sony A65 when first released.  After 2 years and thousands of photos I continue to be impressed & delighted with the photos it produces.  Previously I had four years of very positive experience with the 10 MP Sony A100, which was Sony's first DSLR.When I was ready to upgrade, Sony had proven it's quality.  Also the Sony Alpha series accepts my old Minolta lenses.The higher resolution of the 24 MP A65 was calling to me -- it has delivered stunning results.  The &#34;burst&#34; of 10 images in 1 second was enticing -- and has rewarded me on numerous occasions.  The translucent mirror eliminates a source of noise and vibration, and enables the electronic viewfinder.The electronic viewfinder rules.  It is so brilliant it startled me.  In addition --- it can give you a &#34;heads-up display&#34; of various data including f/stop, shutter speed, ISO setting.  The &#34;EVF&#34; is an outstanding improvement over the traditional  viewfinder.  Another plus: the A65 enables you to zoom in to confirm the autofocus -- or fine tune a manual focus.The A65 has exceeded all my expectations.
I highly recommend the Nikon D3300 because it does everything very well.
I like the camera. It appears very well built and ergonomic. I was surprised though to see that you can only get high
definition in RAW format. In fine definition the average picture is much lower than the 24 mp they claim.
The lens kit is also fine for the price. I came from Canon world and transitioned into the Lumix world for digital.
I think their lenses are better but they do not offer (at least not on mine) the capabilities of a d-SLR camera.
Last thing and the reason for my rating is a terrible customer service. My battery run out in less than 30 days.
Nikon would not replace it as they indicated it was not covered. They would charge $50-$60 for a replacement.
It is not a a good experience and not acceptable from such a reputable company.
"""


i=0
text=sent_tokenize(example_txt)
print(text[2])
aList=[]
aList2=[]
aList3=[]
for x in range(len(text)):
    text2=word_tokenize(text[x])
    a=nltk.pos_tag(text2)
    for x1 in range(len(a)):
        # if [a[x1][1]=="NN" or a[x1][1]=="NNP" or a[x1][1]=="JJ" or a[x1][1]=="ADV"]:
        #     aList3.append(a[x1])
        if a[x1][1]=="NN" or a[x1][1]=="NNP" :
            aList.append(a[x1])
        # if a[x1][1]=="JJ" or a[x1][1]=="ADV":
        #  aList2.append(a[x1])
        x=x+1
    print("\n")
print(aList)
print  np.unique(aList)
print  np.unique(aList2)
#print  np.unique(aList3)
len(a)
query = [e1 for (e1,e2) in a if e2=='NN' or e2=="NNP"]
print(query)

print("The acutal review"+example_txt)
url = "http://text-processing.com/api/sentiment/"

example_updated=(sent_tokenize(example_txt))
sentenced=[]
txt=[]
i=0
for sent in example_updated:
    i=i+1
    sentenced.append(sent)
x=0
pos=0
neg=0
positive=[]
negative=[]
zoom_sentenced=[]
price_sentenced=[]
shutter_sentenced=[]
lens_sentenced=[]
picture_sentenced=[]
battery_sentenced=[]
customer_review=[]

print "\n Sentences related to zoom \n"
for j in range(i):
    if "zoom" in sentenced[j]:
        b=TextBlob(sentenced[j])
        print("The Sentiment using TextBlob is \t")
        print(sentenced[j])
        print(b.sentiment)
        zoom_sentenced.append(sentenced[j])

testapi.app(zoom_sentenced,j)

print("\n Sentences related to price \n")
for j in range(i):
    if "price" in sentenced[j]:
        b=TextBlob(sentenced[j])
        print(sentenced[j])
        print(b.sentiment)
        price_sentenced.append(sentenced[j])
testapi.app(price_sentenced,j)

print("\n Sentences related to Shutter speed \n")
for j in range(i):
     if "shutter" in sentenced[j] and  "zoom" not in sentenced[j]:
        b=TextBlob(sentenced[j])
        print(sentenced[j])
        print(b.sentiment)
        shutter_sentenced.append(sentenced[j])
testapi.app(shutter_sentenced,j)

print("\n Sentences related to lens \n")
for j in range(i):
     if "lens" in sentenced[j] or "wide angle" in sentenced[j]:
        b=TextBlob(sentenced[j])
        print(sentenced[j])
        print(b.sentiment)
        lens_sentenced.append(sentenced[j])
testapi.app(lens_sentenced,j)

print("\n Sentences related to Picture quality \n")
for j in range(i):
     if "high definition" in sentenced[j] or "picture" in sentenced[j]:
        b=TextBlob(sentenced[j])
        print(sentenced[j])
        print(b.sentiment)
        picture_sentenced.append(sentenced[j])
testapi.app(picture_sentenced,j)

print("\n Sentences related to Battery\charger \n")
for j in range(i):
     if "battery" in sentenced[j] or "charge" in sentenced[j]:
        b=TextBlob(sentenced[j])
        print(sentenced[j])
        print(b.sentiment)
        battery_sentenced.append(sentenced[j])
testapi.app(battery_sentenced,j)

print("\n Sentences related to Customer Service \n")
for j in range(i):
     if "customer service" in sentenced[j]:
        b=TextBlob(sentenced[j])
        print(sentenced[j])
        print(b.sentiment)
        customer_review.append(sentenced[j])
testapi.app(customer_review,j)
