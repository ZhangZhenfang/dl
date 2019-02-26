package peer.afang.dl.test;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/2/8 11:26
 */
public class FCLayer {

    private Mat input;
    private Mat weight;
    private Mat out;
    private double bias;
    private Mat delta;
    private Mat grad;
    private Mat inputDelta;
    public FCLayer(int inputSize, int outSize) {
        this.weight = new Mat(inputSize, outSize, CvType.CV_32F);
        this.out = new Mat(outSize, outSize, CvType.CV_32F);
        this.grad = new Mat(inputSize, outSize, CvType.CV_32F);
        double[] data = new double[this.weight.rows() * this.weight.cols()];
        for (int i= 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        this.weight.put(0, 0, data);
        delta = new Mat(1, 1, CvType.CV_32F);
    }



    public void computeOut() {
        Core.gemm(input.reshape(1, 1), weight, 1, new Mat(), 1, out);
    }

    public void computeGrad(Mat label) {
        Mat d1 = new Mat();
        System.out.println(label);
        System.out.println(out);
        Core.subtract(label, out, d1);
        Mat one = Mat.ones(out.size(), CvType.CV_32F);
        Mat d2 = new Mat();
        Core.subtract(one, out, d2);
        Mat d3 = new Mat();
        Core.multiply(out, d2, d3);

        Core.multiply(d3, d1, delta);
        Core.gemm(input.reshape(1, 1).t(), delta, 1, new Mat(), 1, grad);

        System.out.println(input.dump());
        Mat d4 = new Mat();
        one = Mat.ones(input.reshape(1, 1).t().size(), input.type());
        Core.subtract(one, input.reshape(1, 1).t(), d4);
        Mat d5 = new Mat();
        Core.gemm(weight, delta, 1, new Mat(), 1, d5);
        System.out.println(d5.dump());
        Mat d6 = new Mat();
        Core.multiply(input.reshape(1, 1).t(), d4, d6);
        Mat d7 = new Mat();
        Core.multiply(d6, d5, d7);
        System.out.println(d7.dump());
        this.inputDelta = d7;
    }

    public void update(double rate) {
        Mat dst = new Mat();
        Core.multiply(grad, new Scalar(rate), dst);
        Core.add(weight, dst, weight);
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

    public Mat getOut() {
        return out;
    }

    public void setOut(Mat out) {
        this.out = out;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public Mat getDelta() {
        return delta;
    }

    public void setDelta(Mat delta) {
        this.delta = delta;
    }

    public Mat getGrad() {
        return grad;
    }

    public void setGrad(Mat grad) {
        this.grad = grad;
    }

    public Mat getInputDelta() {
        return inputDelta;
    }

    public void setInputDelta(Mat inputDelta) {
        this.inputDelta = inputDelta;
    }
}
