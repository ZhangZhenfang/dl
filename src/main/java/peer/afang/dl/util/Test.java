package peer.afang.dl.util;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.Random;
import java.util.Scanner;

/**
 * @author ZhangZhenfang
 * @date 2019/2/3 9:38
 */
public class Test {
    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }
    static double[] x1 = new double[]{0, 0, 0, -1, -1, 0, -1, 0, -1, 0, -1, -1, 1, 0, -1, -1};
    static double[] x2 = new double[]{0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,1, 0, 0, -1};
    static double[] x3 = new double[]{0, 0, 0, -1, 0, 0, -1, 0, -1, 0, 1, 1, 1, 0, -1, 1};
    static double[] x4 = new double[]{0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1};
    static double[][] labels = new double[][]{{-1.42889927219}, {-0.785398163397}, {0}, {1.46013910562}};
    static double[][] images = new double[][]{x1, x2, x3, x4};

    static Mat w1 = new Mat(2, 2, CvType.CV_32F);
    static Mat w2 = new Mat(2, 2, CvType.CV_32F);
    static Mat w3 = new Mat(4, 1, CvType.CV_32F);
    static double rate = 0.1;
    public static void init() {
        double[] data1 = new double[]{-0.17385154, 2.80282212, -3.52931765, 2.07933109};
        double[] data2 = new double[]{-5.08057429, -2.24798369, 1.37728729, 6.35670686};
        double[] data3 = new double[]{4.13493441, -0.14675518, 0.02265047, 2.38585956};
        Random random = new Random();
        double[] data = new double[4];

        for (int i = 0; i < 4; i++) {
            data[i] = random.nextDouble();
        }
        w1.put(0, 0, data1);
        for (int i = 0; i < 4; i++) {
            data[i] = random.nextDouble();
        }
        w2.put(0, 0, data2);
        for (int i = 0; i < 4; i++) {
            data[i] = random.nextDouble();
        }
        w3.put(0, 0, data3);
        System.out.println(w1.dump());
        System.out.println(w2.dump());
        System.out.println(w3.dump());
    }
    static Mat[] imageMatrix = new Mat[4];
    static Mat[] labelMatrix = new Mat[4];
    public static void main(String[] args) {
        init();

        for (int i = 0; i < 4; i++) {
            Mat image = new Mat(4, 4, CvType.CV_32F);
            Mat label = new Mat(1, 1, CvType.CV_32F);
            image.put(0, 0, images[i]);
            label.put(0, 0, labels[i]);
            imageMatrix[i] = image;
            labelMatrix[i] = label;
            System.out.println(image.dump());
            System.out.println(label.dump());
        }
        double totlaError = 0;
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 4; j++) {
                Mat currentImage = imageMatrix[j];
                Mat currentLabel = labelMatrix[j];
                Mat ttt = new Mat();
                Core.rotate(w1, ttt, 1);
                Mat l1 = MatUtils.conv(currentImage, ttt, 1, 0, 0);
                Mat l1A = new Tanh().activate(l1);
                System.out.println("l1A : \n" + l1A.dump());
                ttt = new Mat();
                Core.rotate(w2, ttt, 1);
                Mat l2 = MatUtils.conv(l1A, ttt, 1, 0, 0);
                System.out.println("l2 :\n" + l2.dump());
                Mat l2A = new Arctan().activate(l2);

                Mat l3IN = l2A.reshape(1, 1);
                Mat l3 = new Mat(1, 1, CvType.CV_32F);
                Core.gemm(l3IN, w3, 1, new Mat(), 1, l3);

                Mat l3A = new Arctan().activate(l3);
                System.out.println("l3A : \n" + l3A.dump());
                Mat t = new Mat(l3A.size(), CvType.CV_32F);
                Core.subtract(l3A, currentLabel, t);
                Core.multiply(t, t, t);

                double cost = MatUtils.sumMat(t) * 0.5;
                System.out.println(cost);
                totlaError += cost;

                Mat grad3pard1 = new Mat();
                Core.subtract(l3A, currentLabel, grad3pard1);
                Mat grad3part2 = new Arctan().derivative(l3);
                Mat grad3part3 = l3IN;
                Mat t1 = new Mat();
                Core.multiply(grad3pard1, grad3part2, t1);
                Mat grad3 = new Mat();
                Core.gemm(grad3part3.t(), t1, 1, new Mat(), 1, grad3);
                System.out.println("grad3:\n" + grad3.dump());


                Mat grad2pardIn = new Mat();
                Core.gemm(t1, w3.t(), 1, new Mat(), 1, grad2pardIn);
                grad2pardIn = grad2pardIn.reshape(1, 2);
                Mat grad2part1 = grad2pardIn;
                Mat grad2part2 = new Arctan().derivative(l2);
                Mat grad2part3 = l1A;
                Mat grad2 = new Mat();
                Mat t2 = new Mat();
                Core.multiply(grad2part1, grad2part2, t2);
                Mat t3 = new Mat();
                Core.rotate(t2, t3, 1);
                ttt = new Mat();
                Core.rotate(t3, ttt, 1);
                Mat t4 = MatUtils.conv(grad2part3, ttt, 1, 0, 0);
                Core.rotate(t4, grad2, 1);
                System.out.println(grad2.dump());



                Mat grad1partInpadweight = MatUtils.paddingZeor(w2, 1, 1, 1, 1);
                Mat t5 = new Mat();
                Core.multiply(grad2part1, grad2part2, t5);
                Mat grad1partIn = new Mat();
                Core.rotate(t5, grad1partIn, 1);




                ttt = new Mat();
                Core.rotate(grad1partIn, ttt, 1);
                Mat grad1part1 = MatUtils.conv(grad1partInpadweight, ttt, 1, 0, 0);
                Mat grad1part2 = new Tanh().derivative(l1);
                System.out.println(l1.dump());
                System.out.println(grad1part1.dump());
                System.out.println("grad1part2\n" + grad1part2.dump());
                Mat grad1part3 = currentImage;
                Mat t6 = new Mat();
                Core.multiply(grad1part1, grad1part2, t6);
                System.out.println(t6.dump());
                Mat t7 = new Mat();
                Core.rotate(t6, t7, 1);
                ttt = new Mat();
                Core.rotate(t7, ttt, 1);
                Mat t8 = MatUtils.conv(grad1part3, ttt, 1, 0, 0);
                Mat grad1 = new Mat();
                Core.rotate(t8, grad1, 1);
//                System.out.println(grad1.dump());
//                new Scanner(System.in).nextLine();

                Mat grad11 = new Mat();
                Core.multiply(grad1, new Scalar(0.1), grad11);
                Core.subtract(w1, grad11, w1);
                Mat grad22 = new Mat();
                Core.multiply(grad2, new Scalar(0.1), grad22);
                Core.subtract(w2, grad22, w2);
                Mat grad33 = new Mat();
                Core.multiply(grad3, new Scalar(0.1), grad33);
                Core.subtract(w3, grad33, w3);
            }
        }
        predict();
    }
    public static void predict() {
        for (int j = 0; j < 4; j++) {
            Mat currentImage = imageMatrix[j];
            Mat currentLabel = labelMatrix[j];
            Mat ttt = new Mat();
            Core.rotate(w1, ttt, 1);
            Mat l1 = MatUtils.conv(currentImage, ttt, 1, 0, 0);
            Mat l1A = new Tanh().activate(l1);
            ttt = new Mat();
            Core.rotate(w2, ttt, 1);
            Mat l2 = MatUtils.conv(l1A, ttt, 1, 0, 0);
            Mat l2A = new Arctan().activate(l2);

            Mat l3IN = l2A.reshape(1, 1);
            Mat l3 = new Mat(1, 1, CvType.CV_32F);
            Core.gemm(l3IN, w3, 1, new Mat(), 1, l3);

            Mat l3A = new Arctan().activate(l3);
            System.out.println("l3A : \n" + l3A.dump());

        }
    }
}
