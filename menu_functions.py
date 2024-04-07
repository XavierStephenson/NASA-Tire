import cv2

def ClickMenu(window_name, title, options, center):
    #Initializing

    title_scale = 2 *(1/.9)

    options_scale = .75
    options_dist = 20
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    img = cv2.imread('Images/MenuBlank.png')
    height, width, c = img.shape
    option_i = [-1]
    final = ['NOTHING']
    scroll = [0]

    #Get all the widths and heights of Text that will be on screen
    options_width = []
    options_height = []
    title_width = 10**9
    while title_width > width:
        title_scale *= .9
        (title_width, title_height) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 1)[0]
    for i in range(len(options)):
        options_width.append(cv2.getTextSize(options[i], cv2.FONT_HERSHEY_SIMPLEX, options_scale, 1)[0][0])
        options_height.append(cv2.getTextSize(options[i], cv2.FONT_HERSHEY_SIMPLEX, options_scale, 1)[0][1])

    #Get the total height to vertically align all the text
    tot_height = title_height + sum(options_height) + len(options)*options_dist
    title_x = int(width/2-title_width/2)
    if tot_height > height:
        offset_to_center = 5
    else:
        offset_to_center = int((height-tot_height)/2)

    #Runs when click or pressed enter
    def Select():
        #If mouse is on a clickable spot, Call that function and destroy menu window
        if option_i[0] < len(options) and option_i[0] >= 0:
            cv2.destroyWindow(window_name)
            final[0] = option_i[0]+1

    #on mouse anything
    def on_click(event, x, y, p1, p2):
        #Clears page, puts title
        img = cv2.imread('Images/MenuBlank.png')
        cv2.putText(img, title, (title_x, title_height+offset_to_center), cv2.FONT_HERSHEY_SIMPLEX , title_scale, (0, 0, 0), 1, cv2.LINE_AA) 

        #event = 10 when 2 fingers on mouse and moves, up when top half down when bottom,
        if event == 10:
            if y<height/2:
                scroll[0] += options_height[0]/5
            else:
                scroll[0] += -options_height[0]/5
            scroll[0] = int(scroll[0])

            #stops from scrolling past tex reg position and from leaving all text of screen
            if scroll[0] > 0: scroll[0] = 0
            if scroll[0] < -tot_height - offset_to_center + options_height[0]: scroll[0] = -tot_height - offset_to_center + options_height[0]

        #Gets postion of mouse relative to the options
        option_i[0] = (y-scroll[0]-title_height-offset_to_center-options_height[0]) // (options_height[0]+options_dist)
        
        #If mouse is on one of the options make a grey rectangle around that option
        if option_i[0] < len(options) and option_i[0] >= 0:
            rect_y = int( (option_i[0]+.5)*(options_height[0]+options_dist)+title_height+offset_to_center ) + scroll[0]
            if center:
                rect_x = int(width/2 - options_width[option_i[0]]/2)
            else:
                rect_x = 0
            cv2.rectangle(img, (rect_x, rect_y), (rect_x+options_width[option_i[0]], rect_y+options_height[0]+5), (100,100,100), -1)

        #Write the text for Each Option, this is after the box so it can be ontop
        for i in range(len(options)):    
            if center:
                rect_x = int(width/2 - options_width[i]/2)
            else:
                rect_x = 0
            rect_y = (i+1)*(options_height[0]+options_dist)+title_height+offset_to_center + scroll[0]
            cv2.putText(img, options[i], (rect_x, rect_y), cv2.FONT_HERSHEY_SIMPLEX , options_scale, (0, 0, 0), 1, cv2.LINE_AA) 
        #Show window
        cv2.imshow(window_name, img)

        #If click
        if event == 4:
            Select()
    #Sets uo mouse event, calls it to set up the text
    cv2.setMouseCallback(window_name, on_click)  
    on_click(0,0,0,0,0)

    while True:
        key = cv2.waitKey(1)
        #If Enter register select
        if key == 13: Select()
        #If esc q or backspace, end program
        if key in (27, ord("q"), 8):
            cv2.destroyWindow(window_name)
            return False
        if final[0] != 'NOTHING':
            return final[0]


def TypeMenu(window_name, msg1, msg2):
    #Initilizing
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    img = cv2.imread('Images/blank.png')
    cv2.putText(img, msg1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, msg2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.imshow(window_name, img)
    name = ''
    while True:
        #Clear Image value, get key press
        img = cv2.imread('Images/blank.png')    
        key = cv2.waitKey(1)
        
        #If no key was press skip all other stuff
        if key == -1: continue
        #If backspace was pressed remove last charcter
        if key == 8:
            name = name[:-1]
        #else add char
        else:
            name += chr(key)

        #Add updated name to img value
        cv2.putText(img, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
        
        #pressed enter button
        if key == 13 :
            #due to key press the name has the enter char appended, remove it
            name = name[:-1]
            return name
            #display img val to screen
        cv2.imshow(window_name, img)


