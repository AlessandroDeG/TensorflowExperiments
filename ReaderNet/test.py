
import time

arrow=[]

for t in range(0,10):
    arrow.append("-")
    


for t in range(0,105):
    if(t%10==0):
        if(t//10 != 0 ):    arrow[(t//10)-1]="-"
        arrow[t//10]=">"
        print("0%" , end = "")
        for s in arrow:
            print(s, end=""), 
        print("100%\r", end=""),
    time.sleep(0.1)
        
