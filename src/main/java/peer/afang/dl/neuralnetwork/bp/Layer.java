package peer.afang.dl.neuralnetwork.bp;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/1/29 14:45
 */
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
            data[i] = (new Random().nextDouble() - 0.5) * 2;
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