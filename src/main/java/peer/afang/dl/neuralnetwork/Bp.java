package peer.afang.dl.neuralnetwork;

import org.opencv.core.*;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/1/20 0:21
 */
public class Bp {
    static {
        String path = "/usr/local/share/OpenCV/java/libopencv_java341.so";
        System.load(path);
    }
    private Mat inputLayer;
    private Mat hiddenLayer;
    private Mat outputLayer;
    private Mat labels;

    private Mat weights1;
    private Mat weights2;

    public static void main(String[] args) {
        double[] input = new double[]{0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1};
        double[] labels = new double[]{0,1, 1, 0};
        Bp bp = new Bp(3, 4, input, labels, 4, 1);
        bp.train(10000, 0.1);
        bp.forward(bp.inputLayer.row(1));
        System.out.println(bp.outputLayer.dump());
        bp.forward(bp.inputLayer.row(0));
        System.out.println(bp.outputLayer.dump());
        bp.forward(bp.inputLayer.row(3));
        System.out.println(bp.outputLayer.dump());
        bp.forward(bp.inputLayer.row(2));
        System.out.println(bp.outputLayer.dump());
    }

    /**
     * @param inputLength 输入向量的长度
     * @param inputNum 用于训练的向量的个数
     * @param hidden 隐藏层向量长度
     * @param out 输出层向量长度
     */
    public Bp(int inputLength, int inputNum, double[] input, double[] labels, int hidden, int out) {
        this.inputLayer = new Mat(inputNum, inputLength, CvType.CV_32F);
        this.inputLayer.put(0, 0, input);
        this.hiddenLayer = new Mat(hidden, 1, CvType.CV_32F);
        this.outputLayer = new Mat(out, 1, CvType.CV_32F);
        this.weights1 = new Mat(inputLength, hidden, CvType.CV_32F);
        this.weights2 = new Mat(hidden, out, CvType.CV_32F);
        this.labels = new Mat(inputNum, 1, CvType.CV_32F);

        double[] v = new double[inputLength * hidden];
        for (int i = 0; i < v.length; i++) {
            v[i] = new Random().nextDouble();
        }
        this.weights1.put(0, 0, v);
        v = new double[hidden * out];
        for (int i = 0; i < v.length; i++) {
            v[i] = new Random().nextDouble();
        }
        this.weights2.put(0, 0, v);
        this.labels.put(0, 0, labels);
        print();
    }

    /**
     * 向前计算求出out
     * @param input
     */
    public void forward(Mat input) {
        // 计算隐藏层
        Core.gemm(input, this.weights1, 1, new Mat(), 1, this.hiddenLayer);
        sigmoid(this.hiddenLayer);
        // 计算输出层
        Core.gemm(this.hiddenLayer, this.weights2, 1, new Mat(), 1, this.outputLayer);
        sigmoid(this.outputLayer);
    }

    /**
     * 反向传播更新权值
     * @param input
     * @param lable
     */
    public void backPropagation(Mat input, Mat lable, double rate) {
        // 对应输出层节点误差delta公式，以及权值更新公式
        Mat one = Mat.ones(outputLayer.size(), CvType.CV_32F);
        Mat t1 = new Mat(outputLayer.size(), CvType.CV_32F);
        Mat t2= new Mat(outputLayer.size(), CvType.CV_32F);
        Mat deltaOut = new Mat(outputLayer.size(), CvType.CV_32F);
        Core.subtract(one, outputLayer, t1);
        Core.multiply(outputLayer, t1, t1);
        Core.subtract(lable, outputLayer, t2);
        Core.multiply(t1, t2, deltaOut);
        Mat t3 = new Mat(weights2.size(), CvType.CV_32F);
        Core.gemm(hiddenLayer.t(), deltaOut.t(), 1, new Mat(), 1, t3);
        Core.multiply(t3, new Scalar(rate), t3);
        Core.add(t3, weights2, weights2);

        // 对应隐藏层节点误差公式，以及权值更新公式
        Mat deltaHidden = new Mat(hiddenLayer.size(), CvType.CV_32F);
        Core.gemm(weights2, deltaOut, 1, new Mat(), 1, deltaHidden);
        one = Mat.ones(hiddenLayer.size(), CvType.CV_32F);
        Mat t5 = new Mat(hiddenLayer.size(), CvType.CV_32F);
        Core.subtract(one, hiddenLayer, t5);
        Core.multiply(hiddenLayer, t5, t5);
        Core.multiply(t5, deltaHidden.t(), t5);
        Mat t6 = new Mat(weights1.size(), CvType.CV_32F);
        Core.gemm(input.t(), deltaHidden.t(), 1, new Mat(), 1, t6);
        Core.multiply(t6, new Scalar(rate), t6);
        Core.add(weights1, t6, weights1);
    }


    public void print() {
        System.out.println("input:\n" + inputLayer.dump());
        System.out.println("hide:\n" + hiddenLayer.dump());
        System.out.println("out:\n" + outputLayer.dump());
        System.out.println("lables:\n" + labels.dump());
        System.out.println("weights1:\n" + weights1.dump());
        System.out.println("weights2:\n" + weights2.dump());
    }

    /**
     * 对网络进行训练
     * @param times 训练次数
     * @param rate 学习速率
     */
    public void train(int times, double rate) {
        for (int i = 0; i < times; i++) {
            for (int j = 0; j < inputLayer.rows(); j++) {
                forward(this.inputLayer.row(j));
                backPropagation(this.inputLayer.row(j), this.labels.row(j), rate);
            }
        }
    }

    /**
     * 对矩阵的每一个元素进行sigmoid运算
     * @param m
     */
    public void sigmoid(Mat m) {
        for (int i = 0; i < m.cols(); i++) {
            // 取值经过sigmoid后再放回原处
            m.put(0, i, new double[]{sigmoidFunction(m.get(0, i)[0])});
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
}
