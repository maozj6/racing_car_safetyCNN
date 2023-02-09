
import numpy as np

if __name__ == '__main__':
    label=np.load("/home/mao/23Spring/cars/racing_car_data/record/train/labels.npy")
    serise=np.load("/home/mao/23Spring/cars/racing_car_data/record/train/serise.npy")
    imgs=[]
    labels=[]
    leng=20
    guard = 0
    for i in range(len(serise)):

        for j in range(serise[i]):
            guard = guard + 1
            if(j+leng<serise[i]):
                imgs.append(guard)

                labels.append(guard+leng)

        # if(serise[i])!=950:
        #     print(i)
        #     print(serise[i])
    imgs=np.array(imgs)
    labels=np.array(labels)
    print(imgs[931])
    print(labels[931])
    print(imgs[950])
    print(labels[950])
    print(serise)
    print(len(imgs))
    print(len(labels))

    print("end")
