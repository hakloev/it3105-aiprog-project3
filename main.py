from module5.mnist import mnist_basics
import matplotlib.pyplot as pyplot

if __name__ == "__main__":
    #  mnist_basics.quicktest()
    #  mnist_basics.show_avg_digit(3)

    train_set_images, train_set_labels = mnist_basics.load_all_flat_cases(type="training")
    test_set_images, test_set_labels = mnist_basics.load_all_flat_cases(type="testing")



    #  pyplot.show(block=True)

