package peer.afang.dl.neuralnetwork;

import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * @author ZhangZhenfang
 * @date 2019/1/20 0:21
 */
public class Bp {
    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }
    private Mat outputLayer;
    private List<Mat> weights;
    private List<Mat> hiddenLayers;
    private Mat delta;
    public static void main(String[] args) {
        double[][] input = new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}};
        double[] labels = new double[]{0,1, 1, 0};
        Bp bp = new Bp(3, 1, new int[]{3}, 1);
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < input.length; j++) {
                Mat m = new Mat(1, 3, CvType.CV_32F);
                m.put(0, 0, input[j]);
                bp.newForward(m);
                bp.print();
                new Scanner(System.in).nextLine();
                Mat label = new Mat(1, 1, CvType.CV_32F);
                label.put(0, 0, new double[]{labels[j]});
                bp.newBackPropagation(m, label, 0.1);

            }
        }
        for (int j = 0; j < input.length; j++) {
            Mat m = new Mat(1, 3, CvType.CV_32F);
            m.put(0, 0, input[j]);
            bp.newForward(m);
            System.out.println(bp.outputLayer.dump());
        }
    }

    /**
     * @param inputLength 输入程向量长度
     * @param hiddeNum 隐藏层层数
     * @param hiddeLengthes 隐藏层各层向量长度
     * @param outLength 输出层向量长度
     */
    public Bp(int inputLength, int hiddeNum, int[] hiddeLengthes, int outLength) {
        this.weights = new ArrayList<Mat>();
        this.hiddenLayers = new ArrayList<Mat>();
        int row = inputLength;
        for (int i = 0; i < hiddeNum; i++) {
            Mat m = new Mat(row, hiddeLengthes[i], CvType.CV_32F);
            this.weights.add(m);
            double[] v = new double[m.rows() * m.cols()];
            for (int j = 0; j < v.length; j++) {
                v[j] = 0;//(new Random().nextDouble() - 0.5) * 2;
            }
            m.put(0, 0, v);
            row = hiddeLengthes[i];
            m = new Mat(hiddeLengthes[i], 1, CvType.CV_32F);
            this.hiddenLayers.add(m);
        }
        Mat m = new Mat(row, outLength, CvType.CV_32F);
        double[] v = new double[m.rows() * m.cols()];
        for (int j = 0; j < v.length; j++) {
            v[j] = 0;//(new Random().nextDouble() - 0.5) * 2;
        }
        m.put(0, 0, v);
        weights.add(m);
        this.outputLayer = new Mat(outLength, 1, CvType.CV_32F);
    }

    /**
     * 向前计算求出out
     * @param input
     */
    public void newForward(Mat input) {
        for (int i = 0; i < this.weights.size() - 1; i++) {
            // 计算隐藏层
            Mat t = new Mat(this.hiddenLayers.get(i).t().size(), CvType.CV_32F);
            Core.gemm(input, this.weights.get(i), 1, new Mat(), 1, t);
            Core.transpose(t, this.hiddenLayers.get(i));
            sigmoid(this.hiddenLayers.get(i));
            input = this.hiddenLayers.get(i).t();
        }
        // 计算输出层
        Core.gemm(this.hiddenLayers.get(this.hiddenLayers.size() - 1).t(), this.weights.get(this.weights.size() - 1),
                1, new Mat(), 1, this.outputLayer);
        this.outputLayer = this.outputLayer.t();
        sigmoid(this.outputLayer);
    }

    /**
     * 反向传播更新权值
     * @param input
     * @param lable
     */
    public void newBackPropagation(Mat input, Mat lable, double rate) {
        // 对应输出层节点误差delta公式，以及权值更新公式
        Mat one = Mat.ones(outputLayer.size(), CvType.CV_32F);
        Mat t1 = new Mat(outputLayer.size(), CvType.CV_32F);
        Mat t2= new Mat(outputLayer.size(), CvType.CV_32F);
        Mat deltaOut = new Mat(outputLayer.size(), CvType.CV_32F);
        Core.subtract(one, outputLayer, t1);
        Core.multiply(outputLayer, t1, t1);
        Core.subtract(lable, outputLayer, t2);
        Core.multiply(t1, t2, deltaOut);
        Mat t3 = new Mat(this.weights.get(this.weights.size() - 1).size(), CvType.CV_32F);
        Core.gemm(hiddenLayers.get(hiddenLayers.size() - 1), deltaOut.t(), 1, new Mat(), 1, t3);
        Core.multiply(t3, new Scalar(rate), t3);
        Core.add(t3, this.weights.get(this.weights.size() - 1), this.weights.get(this.weights.size() - 1));

        for (int i = this.weights.size() - 2; i >= 1; i--) {
            // 对应隐藏层节点误差公式，以及权值更新公式
            Mat deltaHidden = new Mat(hiddenLayers.get(i).size(), CvType.CV_32F);
            Mat t5 = new Mat(hiddenLayers.get(i).size(), CvType.CV_32F);
            Mat t6 = new Mat(hiddenLayers.get(i).size(), CvType.CV_32F);
            Core.gemm(weights.get(i + 1), deltaOut, 1, new Mat(), 1, t5);
            one = Mat.ones(hiddenLayers.get(i).size(), CvType.CV_32F);
            Core.subtract(one, hiddenLayers.get(i), t6);
            Core.multiply(hiddenLayers.get(i), t6, t6);
            Core.multiply(t6, t5, deltaHidden);
            Mat t7 = new Mat(weights.get(i).size(), CvType.CV_32F);
            Core.gemm(hiddenLayers.get(i - 1), deltaHidden.t(), 1, new Mat(), 1, t7);
            Core.multiply(t7, new Scalar(rate), t7);
            Core.add(weights.get(i), t7, weights.get(i));
            deltaOut = deltaHidden;
        }
        // 对应隐藏层节点误差公式，以及权值更新公式
        Mat deltaHidden = new Mat(hiddenLayers.get(0).size(), CvType.CV_32F);
        Mat t5 = new Mat(hiddenLayers.get(0).size(), CvType.CV_32F);
        Mat t6 = new Mat(hiddenLayers.get(0).size(), CvType.CV_32F);
        Core.gemm(weights.get(1), deltaOut, 1, new Mat(), 1, t5);
        one = Mat.ones(hiddenLayers.get(0).size(), CvType.CV_32F);
        Core.subtract(one, hiddenLayers.get(0), t6);
        Core.multiply(hiddenLayers.get(0), t6, t6);
        Core.multiply(t6, t5, deltaHidden);
        Mat t7 = new Mat(weights.get(0).size(), CvType.CV_32F);
        Core.gemm(input.t(), deltaHidden.t(), 1, new Mat(), 1, t7);
        Core.multiply(t7, new Scalar(rate), t7);
        Core.add(weights.get(0), t7, weights.get(0));
        this.delta = deltaHidden;
    }

    public void print() {
        System.out.println("hides:\n");
        for (Mat m : hiddenLayers) {
            System.out.println(m.dump());
        }
        System.out.println("out:\n" + outputLayer.dump());
        System.out.println("weights\n");
        for (Mat m : weights) {
            System.out.println(m.dump());
        }
    }

    /**
     * 对矩阵的每一个元素进行sigmoid运算
     * @param m
     */
    public void sigmoid(Mat m) {
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                // 取值经过sigmoid后再放回原处
                m.put(i, j, new double[]{sigmoidFunction(m.get(i, j)[0])});
            }
        }
    }

    /**
     * sigmoid激活函数
     * @param x 输入值
     * @return
     */
    public double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public Mat getOutputLayer() {
        return outputLayer;
    }

    public void setOutputLayer(Mat outputLayer) {
        this.outputLayer = outputLayer;
    }

    public Mat getDelta() {
        return delta;
    }

    public void setDelta(Mat delta) {
        this.delta = delta;
    }
}
