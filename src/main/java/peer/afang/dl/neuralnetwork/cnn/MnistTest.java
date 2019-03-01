package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import peer.afang.dl.util.*;

import java.util.Random;
import java.util.Scanner;

/**
 * @author ZhangZhenfang
 * @date 2019/2/28 18:42
 */
public class MnistTest {
    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }
    public static void main(String[] args) {
        Mat[] inputs = new Mat[200];
        Mat[] labels = new Mat[200];
        Mat[] weights = new Mat[6];

        MnistReader reader = new MnistReader(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
        for (int i = 0; i < inputs.length; i++) {
            double[] nextImage = reader.getNextImage(true);
            double[] nextLabel = reader.getNextLabel();
            Mat m = new Mat(28, 28, CvType.CV_32F);
            m.put(0, 0, nextImage);
            Mat l = new Mat(1, 10, CvType.CV_32F);
            l.put(0, 0, nextLabel);
            inputs[i] = m;
//            System.out.println(inputs[i].dump());
            labels[i] = l;
//            System.out.println(labels[i].dump());
//            new Scanner(System.in).nextLine();
        }
        for (int i = 0; i < 3; i++) {
            Mat m = new Mat(4, 4, CvType.CV_32F);
            double[] data = new double[16];
            for (int j = 0; j < data.length; j++) {
                data[j] = (new Random().nextDouble() - 0.5) * 2;
            }
            m.put(0, 0, data);
            weights[i] = m;
        }
        Mat m = new Mat(361, 150, CvType.CV_32F);
        double[] data = new double[361 * 150];
        for (int i = 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        m.put(0, 0, data);
        weights[3] = m;
        m = new Mat(150, 150, CvType.CV_32F);
        data = new double[150 * 150];
        for (int i = 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        m.put(0, 0, data);
        weights[4] = m;

        m = new Mat(150, 10, CvType.CV_32F);
        data = new double[150 * 10];
        for (int i = 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        m.put(0, 0, data);
        weights[5] = m;

//        for (int i = 0; i < weights.length; i++) {
//            System.out.println(weights[i].dump());
//        }
//        new Scanner(System.in).nextLine();
        double b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
        Activator relu = new Relu();
        Activator sigmoid = new Sigmoid();

        for (int j = 0; j < 50; j++) {

            for (int i = 0; i < inputs.length; i++) {
                // 第一层卷积层
                Mat z1 = MatUtils.conv(inputs[i], weights[0], 1, 0, 0);
                Core.add(z1, new Scalar(b1), z1);
                Mat a1 = relu.activate(z1);

                // 第二层卷积层
                Mat z2 = MatUtils.conv(a1, weights[1], 1, 0, 0);
                Core.add(z2, new Scalar(b2), z2);
                Mat a2 = relu.activate(z2);

                // 第三层卷积层
                Mat z3 = MatUtils.conv(a2, weights[2], 1, 0, 0);
                Core.add(z3, new Scalar(b3), z3);
                Mat a3 = relu.activate(z3);


                // 全连接层四
                Mat z4 = new Mat();
                Core.gemm(a3.reshape(1, 1), weights[3], 1, new Mat(), 1, z4);
                Core.add(z4, new Scalar(b4), z4);
                Mat a4 = sigmoid.activate(z4);

                // 全连接层五
                Mat z5 = new Mat();
                Core.gemm(a4, weights[4], 1, new Mat(), 1, z5);
                Core.add(z5, new Scalar(b5), z5);
                Mat a5 = sigmoid.activate(z5);

                // 全连接层六
                Mat z6 = new Mat();
                Core.gemm(a5, weights[5], 1, new Mat(), 1, z6);
                Core.add(z6, new Scalar(b6), z6);
                Mat a6 = sigmoid.activate(z6);



                // 计算全连接层六delta
                Mat delta6 = new Mat();
                Mat derivativeZ6 = sigmoid.derivative(a6);
                Mat deltaOut = new Mat();
                Core.subtract(labels[i], a6, deltaOut);
                Core.multiply(deltaOut, derivativeZ6, delta6);
                // 计算全连接层六梯度
                Mat grad6 = new Mat();
                Core.gemm(a5.t(), delta6, 1, new Mat(), 1, grad6);


                // 计算全连接层五delta
                Mat delta5 = new Mat();
                Mat derivativeZ5 = sigmoid.derivative(a5);
                Mat t2 = new Mat();
                Core.gemm(weights[5], delta6.t(), 1, new Mat(), 1, t2);
                Core.multiply(t2.t(), derivativeZ5, delta5);
                // 计算全连接层五梯度
                Mat grad5 = new Mat();
                Core.gemm(a4.t(), delta5, 1, new Mat(), 1, grad5);

                // 计算全连接层四delta
                Mat delta4 = new Mat();
                Mat derivativeZ4 = sigmoid.derivative(a4);
//                Mat t2 = new Mat();
                Core.gemm(weights[4], delta5.t(), 1, new Mat(), 1, t2);
                Core.multiply(t2.t(), derivativeZ4, delta4);
                // 计算全连接层四梯度
                Mat grad4 = new Mat();
                Core.gemm(a3.reshape(1, 1).t(), delta4, 1, new Mat(), 1, grad4);


                // 计算卷积层三delta
                Mat delta3 = new Mat();
                Mat derivativeZ3 = relu.derivative(a3);
                Core.gemm(weights[3], delta4.t(), 1, new Mat(), 1, t2);
                Mat t3 = t2.reshape(1, 19);
                Core.multiply(t3, derivativeZ3, delta3);
                // 计算第二层卷积层梯度
                Mat grad3 = MatUtils.conv(a2, delta3, 1, 0, 0);

                // 计算卷积层二delta
                Mat delta2 = new Mat();
                Mat paddedDelta3 = MatUtils.paddingZeor(delta3, 3, 3, 3, 3);
                Mat conv = MatUtils.conv(paddedDelta3, MatUtils.rotate180(weights[2]), 1, 0, 0);
                Mat derivativeZ2 = relu.derivative(a2);
                Core.multiply(conv, derivativeZ2, delta2);
                // 计算卷积层二梯度
                Mat grad2 = MatUtils.conv(a1, delta2, 1, 0, 0);

                // 计算卷积层二delta
                Mat delta1 = new Mat();
                Mat paddedDelta2 = MatUtils.paddingZeor(delta2, 3, 3, 3, 3);
                conv = MatUtils.conv(paddedDelta2, MatUtils.rotate180(weights[1]), 1, 0, 0);
                Mat derivativeZ1 = relu.derivative(a1);
                Core.multiply(conv, derivativeZ1, delta1);
                // 计算第一层卷积层梯度
                Mat grad1 = MatUtils.conv(inputs[i], delta1, 1, 0, 0);

//                System.out.println(delta1.dump());
//                System.out.println(grad1);
//                System.out.println(grad1.dump());
//                new Scanner(System.in).nextLine();


//                // 计算全连接层delta
//                Mat delta3 = new Mat();
//                Mat derivativeZ3 = sigmoid.derivative(a3);
////                Mat deltaOut = new Mat();
//                Core.subtract(labels[i], a3, deltaOut);
//                Core.multiply(deltaOut, derivativeZ3, delta3);
//                // 计算全连接层梯度
//                Mat grad3 = new Mat();
//                Core.gemm(a2.reshape(1, 1).t(), delta3, 1, new Mat(), 1, grad3);
//
                // 计算第二层卷积层delta
//                Mat delta2 = new Mat();
//                Mat derivativeZ2 = relu.derivative(a2);
////                Mat t2 = new Mat();
//                Core.gemm(weights[2], delta3.t(), 1, new Mat(), 1, t2);
////                Mat t3 = t2.reshape(1, 2);
//                Core.multiply(t3, derivativeZ2, delta2);
//                // 计算第二层卷积层梯度
//                Mat grad2 = MatUtils.conv(a1, delta2, 1, 0, 0);
//
//                // 计算第一层卷积层delta
//                Mat delta1 = new Mat();
//                Mat paddedDelta2 = MatUtils.paddingZeor(delta2, 1, 1, 1, 1);
//                Mat conv = MatUtils.conv(paddedDelta2, MatUtils.rotate180(weights[1]), 1, 0, 0);
//                Mat derivativeZ1 = relu.derivative(a1);
//                Core.multiply(conv, derivativeZ1, delta1);
//                // 计算第一层卷积层梯度
//                Mat grad1 = MatUtils.conv(inputs[i], delta1, 1, 0, 0);
                /**
                 * 更新权重和偏执
                 */
                double rate = 0.1;
                Mat mm = new Mat();
                Core.multiply(grad6, new Scalar(rate), mm);
//                System.out.println(weights[5].dump());
                Core.add(weights[5], mm, weights[5]);
                b6 = b6 - rate * MatUtils.sumMat(delta6);
//                System.out.println(grad6.dump());
//                System.out.println(weights[5].dump());
//                new Scanner(System.in).nextLine();

                Core.multiply(grad5, new Scalar(rate), mm);
                Core.add(weights[4], mm, weights[4]);
                b5 = b5 - rate * MatUtils.sumMat(delta5);
                Core.multiply(grad4, new Scalar(rate), mm);
                Core.add(weights[3], mm, weights[3]);
                b4 = b4 - rate * MatUtils.sumMat(delta4);
                Core.multiply(grad3, new Scalar(rate), mm);
                Core.add(weights[2], mm, weights[2]);
                b3 = b3 - rate * MatUtils.sumMat(delta3);
                Core.multiply(grad2, new Scalar(rate), mm);
                Core.add(weights[1], mm, weights[1]);
                b2 = b2 - rate * MatUtils.sumMat(delta2);
                Core.multiply(grad1, new Scalar(rate), mm);
                Core.add(weights[0], mm, weights[0]);
                b1 = b1 - rate * MatUtils.sumMat(delta1);
            }
        }

//        System.out.println("weights");
//        System.out.println(weights[0].dump());
//        System.out.println(b1);
//        System.out.println(weights[1].dump());
//        System.out.println(b2);
//        System.out.println(weights[2].dump());
//        System.out.println(b3);
//        System.out.println("\n\n");

        for (int i = 0; i < weights.length; i++) {
            System.out.println(weights[i].dump());
        }
        // 测试
        for (int i = 0; i < inputs.length; i++) {
            System.out.println(inputs[i].dump());
            // 第一层卷积层
            Mat z1 = MatUtils.conv(inputs[i], weights[0], 1, 0, 0);
            Core.add(z1, new Scalar(b1), z1);
            Mat a1 = relu.activate(z1);

            System.out.println("a1");
            System.out.println(a1.dump());
            // 第二层卷积层
            Mat z2 = MatUtils.conv(a1, weights[1], 1, 0, 0);
            Core.add(z2, new Scalar(b2), z2);
            Mat a2 = relu.activate(z2);
            System.out.println("a2");
            System.out.println(a2.dump());
            // 第三层卷积层
            Mat z3 = MatUtils.conv(a2, weights[2], 1, 0, 0);
            Core.add(z3, new Scalar(b3), z3);
            Mat a3 = relu.activate(z3);
            System.out.println("a3");
            System.out.println(a3.dump());

            // 全连接层四
            Mat z4 = new Mat();
            Core.gemm(a3.reshape(1, 1), weights[3], 1, new Mat(), 1, z4);
            Core.add(z4, new Scalar(b4), z4);
            Mat a4 = sigmoid.activate(z4);
            System.out.println("a4");
            System.out.println(a4.dump());
            // 全连接层五
            Mat z5 = new Mat();
            Core.gemm(a4, weights[4], 1, new Mat(), 1, z5);
            Core.add(z5, new Scalar(b5), z5);
            Mat a5 = sigmoid.activate(z5);
            System.out.println("a5");
            System.out.println(a5.dump());
            // 全连接层六
            Mat z6 = new Mat();
            Core.gemm(a5, weights[5], 1, new Mat(), 1, z6);
            Core.add(z6, new Scalar(b6), z6);
            Mat a6 = sigmoid.activate(z6);
            System.out.println("a6");
            System.out.println(a6.dump());
            System.out.println(labels[i].dump());
            new Scanner(System.in).nextLine();
        }
    }
}
