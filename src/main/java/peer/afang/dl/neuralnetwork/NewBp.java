package peer.afang.dl.neuralnetwork;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * @author ZhangZhenfang
 * @date 2019/1/28 20:11
 */
public class NewBp {

    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }

    public static void main(String[] args) {
        double[][] inputs = new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}};
        double[] labels = new double[]{0,1, 1, 0};
        Mat input = new Mat(1, 3, CvType.CV_32F);
        input.put(0, 0, inputs[0]);
        NewBp newBp = new NewBp(input, 1, new int[]{3}, 1);
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < 4; j++) {
                input = new Mat(1, 3, CvType.CV_32F);
                input.put(0, 0, inputs[j]);
                newBp.forward(input);
                System.out.println(newBp.outLayer.getOutput().dump());
                System.out.println(newBp.hidLayers.get(0).getWeight().dump());
                System.out.println(newBp.hidLayers.get(0).getOutput().dump());
                System.out.println(newBp.outLayer.getWeight().dump());
                new Scanner(System.in).nextLine();
                Mat label = new Mat(1, 1, CvType.CV_32F);
                label.put(0, 0, labels[j]);
                newBp.back(input, label, 0.1);

            }
        }
        Mat in = new Mat(1, 3, CvType.CV_32F);
        in.put(0, 0, inputs[0]);
        newBp.forward(in);
        System.out.println(newBp.outLayer.getOutput().dump());
        in = new Mat(1, 3, CvType.CV_32F);
        in.put(0, 0, inputs[1]);
        newBp.forward(in);
        System.out.println(newBp.outLayer.getOutput().dump());
        in = new Mat(1, 3, CvType.CV_32F);
        in.put(0, 0, inputs[2]);
        newBp.forward(in);
        System.out.println(newBp.outLayer.getOutput().dump());
        in = new Mat(1, 3, CvType.CV_32F);
        in.put(0, 0, inputs[3]);
        newBp.forward(in);
        System.out.println(newBp.outLayer.getOutput().dump());

    }

    private List<HidLayer> hidLayers;
    private OutLayer outLayer;

    public NewBp(Mat input, int numberOfHidLayer, int[] lengths, int outputLength) {
        this.hidLayers = new ArrayList<HidLayer>();
        HidLayer hidLayer = new HidLayer(input, lengths[0]);
        hidLayer.print();
        hidLayers.add(hidLayer);
        for (int i = 1; i < numberOfHidLayer; i++) {
            hidLayer = new HidLayer(hidLayer.getOutput(), lengths[i]);
            hidLayer.print();
            hidLayers.add(hidLayer);
        }
        outLayer = new OutLayer(hidLayer.getOutput(), outputLength);
        outLayer.print();
    }

    public void forward(Mat input) {
        hidLayers.get(0).setInput(input);
        for (int i = 0; i < hidLayers.size(); i++) {
            HidLayer layer = hidLayers.get(i);
            layer.computeOut();
            sigmoid(layer.getOutput());
        }
        outLayer.computeOut();
//        System.out.println("____________________________________________________");
//        System.out.println(outLayer.getOutput().dump());
        sigmoid(outLayer.getOutput());
//        System.out.println(outLayer.getOutput().dump());
    }

    public void back(Mat input, Mat label, double rate) {
        outLayer.computeDelta(label, new Mat());
        outLayer.updateWeight(rate);
        hidLayers.get(hidLayers.size() - 1).computeDelta(outLayer.getDelta(), outLayer.getWeight());
        hidLayers.get(hidLayers.size() - 1).updateWeight(rate);
        for (int i = hidLayers.size() - 2; i >= 0; i--) {
            hidLayers.get(i).computeDelta(hidLayers.get(i + 1).getDelta(), hidLayers.get(i + 1).getWeight());
            hidLayers.get(i).updateWeight(rate);
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
}

abstract class Layer {
    private Mat input;
    private Mat weight;
    private Mat output;
    private Mat delta;
    private Mat one;
    public Layer() {

    }
    public Layer(Mat input, int outputLength) {
        this.input = input;
        this.output = Mat.zeros(input.rows(), outputLength, CvType.CV_32F);
        this.delta = Mat.zeros(input.rows(), outputLength, CvType.CV_32F);
        this.weight = new Mat(input.cols(), outputLength, CvType.CV_32F);
        int length = weight.cols() * weight.rows();
        double[] data = new double[length];
        for (int i = 0; i < length; i++) {
            data[i] = 0;//(new Random().nextDouble() - 0.5) * 2;
        }
        this.weight.put(0, 0, data);
        this.one = Mat.ones(this.output.size(), CvType.CV_32F);
    }

    abstract void computeOut();
    abstract void computeDelta(Mat lastDelta, Mat lastWeight);

    public void updateWeight(double rate) {
        Mat dst = new Mat(weight.size(), CvType.CV_32F);
//        System.out.println("*******************88");
//        System.out.println(input);
//        System.out.println(delta);
        Core.gemm(input.t(), delta, 1, new Mat(), 1, dst);
        Core.multiply(dst, new Scalar(rate), dst);
        Core.add(weight, dst, weight);
    }

    void print() {
        System.out.println("input :");
        System.out.println(input);
        System.out.println("weight :");
        System.out.println(weight);
        System.out.println("output :");
        System.out.println(output);
    }
    public Mat getInput() {
        return input;
    }

    public void setInput(Mat input) {
        this.input = input;
    }

    public Mat getWeight() {
        return weight;
    }

    public void setWeight(Mat weight) {
        this.weight = weight;
    }

    public Mat getOutput() {
        return output;
    }

    public void setOutput(Mat output) {
        this.output = output;
    }

    public Mat getDelta() {
        return delta;
    }

    public void setDelta(Mat delta) {
        this.delta = delta;
    }

    public Mat getOne() {
        return one;
    }

    public void setOne(Mat one) {
        this.one = one;
    }
}

class HidLayer extends Layer {

    public HidLayer(Mat input, int outputLength) {
        super(input, outputLength);
    }
    @Override
    void computeOut() {
        System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        System.out.println(getInput().dump());
        System.out.println(getWeight().dump());
        Core.gemm(this.getInput(), this.getWeight(), 1, new Mat(), 1, this.getOutput());

    }



    @Override
    void computeDelta(Mat lastDelta, Mat lastWeight) {
        Mat dst1 = new Mat(this.getOne().size(), CvType.CV_32F);
        Core.subtract(this.getOne(), this.getOutput(), dst1);
        Mat dst2 = new Mat(this.getOne().size(), CvType.CV_32F);
//        System.out.println(lastWeight);
//        System.out.println(lastDelta);
        Core.gemm(lastWeight, lastDelta.t(), 1, new Mat(), 1, dst2);
        Core.multiply(this.getOutput(), dst1, dst1);
        Core.multiply(dst1, dst2.t(), this.getDelta());
    }
}

class OutLayer extends Layer {

    public OutLayer(Mat input, int outputLength) {
        super(input, outputLength);
    }
    @Override
    void computeOut() {
//        System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
//        System.out.println(getInput().dump());
//        System.out.println(getWeight().dump());
        Core.gemm(this.getInput(), this.getWeight(), 1, new Mat(), 1, this.getOutput());
    }

    @Override
    void computeDelta(Mat label, Mat mat) {
        Mat dst1 = new Mat(this.getOne().size(), CvType.CV_32F);
        Core.subtract(this.getOne(), this.getOutput(), dst1);
        Mat dst2 = new Mat(this.getOne().size(), CvType.CV_32F);
        Core.subtract(label, this.getOutput(), dst2);
        Core.multiply(this.getOutput(), dst1, dst1);
        Core.multiply(dst1, dst2, this.getDelta());
    }
}