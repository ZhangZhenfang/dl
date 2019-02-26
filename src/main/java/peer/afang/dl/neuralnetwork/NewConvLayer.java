package peer.afang.dl.neuralnetwork;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import peer.afang.dl.util.*;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/2/3 17:21
 */
public class NewConvLayer {

    private boolean flag;
    private int weightSize;

    private Mat weight;

    private Mat input;

    private Mat activatedOutput;

    private Mat out;

    private Activator activator;
    private Mat part1;
    private Mat part2;
    private Mat part3;
    private Mat grad;

    public NewConvLayer(int weightSize, int inputSize, Activator activator, boolean flag) {
        this.flag = flag;
        this.activator = activator;
        this.weight = new Mat(weightSize, weightSize, CvType.CV_32F);
        double[] data = new double[weightSize * weightSize];
        int index = 0;
        for (int i = 0; i < data.length; i++) {
            data[index++] = (new Random().nextDouble() - 0.5) * 2;
        }
        weight.put(0, 0, data);
//        System.out.println(weight.dump());
        int outSize = (inputSize + 2 * 0 - weightSize) / 1 + 1;
        this.out = new Mat(outSize, outSize, CvType.CV_32F);
        this.activatedOutput = new Mat(outSize, outSize, CvType.CV_32F);
    }

    public void computeOut() {
        Mat t = new Mat();

        Core.rotate(weight, t, 1);
//        (input.rows() + 2 * 0 - weight.rows()) / 1 + 1;
//        System.out.println(input.dump());
//        System.out.println(t.dump());
        MatUtils.conv(input, t, out, 1, 0, 0);
//        System.out.println("************************************");
//        System.out.println(input.dump());
//        System.out.println(t.dump());
//        System.out.println(out.dump());
        activator.activate(out, activatedOutput);
//        System.out.println(activatedOutput.dump());
    }

    public void computeGrad(Mat lastPart1, Mat lastPart2, Mat lastWeight) {
        if (flag) {
            Mat ttt;
            Mat grad2pardIn = new Mat();
            Mat t1 = new Mat();
            Core.multiply(lastPart1, lastPart2, t1);
            Core.gemm(t1, lastWeight.t(), 1, new Mat(), 1, grad2pardIn);
//            System.out.println(grad2pardIn);
            grad2pardIn = grad2pardIn.reshape(1, 13);
            part1 = grad2pardIn;
            part2 = new Arctan().derivative(out);
            part3 = input;
            Mat grad2 = new Mat();
            Mat t2 = new Mat();
            Core.multiply(part1, part2, t2);
            Mat t3 = new Mat();
            Core.rotate(t2, t3, 1);
            ttt = new Mat();
            Core.rotate(t3, ttt, 1);
            Mat t4 = MatUtils.conv(part3, ttt, 1, 0, 0);
            Core.rotate(t4, grad2, 1);
            grad = grad2;
        } else {
            Mat ttt;
            int padSize = out.rows() - lastWeight.rows();
//            System.out.println("padsize : " + padSize);
            Mat grad1partInpadweight = MatUtils.paddingZeor(lastWeight, padSize, padSize, padSize, padSize);
            Mat t5 = new Mat();
            Core.multiply(lastPart1, lastPart2, t5);
            Mat grad1partIn = new Mat();
            Core.rotate(t5, grad1partIn, 1);

            ttt = new Mat();
            Core.rotate(grad1partIn, ttt, 1);
//            System.out.println(grad1partInpadweight);
//            System.out.println(ttt);
            part1 = MatUtils.conv(grad1partInpadweight, ttt, 1, 0, 0);
            part2 = new Tanh().derivative(out);
            part3 = input;
            Mat t6 = new Mat();
//            System.out.println(part1);
//            System.out.println(part2);
            Core.multiply(part1, part2, t6);
            Mat t7 = new Mat();
            Core.rotate(t6, t7, 1);
            ttt = new Mat();
            Core.rotate(t7, ttt, 1);
            Mat t8 = MatUtils.conv(part3, ttt, 1, 0, 0);
            Mat grad1 = new Mat();
            Core.rotate(t8, grad1, 1);
            grad = grad1;
        }

    }

    public void updateWeight(double rate) {
        Mat tmp = new Mat();
        Core.multiply(grad, new Scalar(rate), tmp);
        Core.subtract(weight, tmp, weight);
    }

    public boolean isFlag() {
        return flag;
    }

    public void setFlag(boolean flag) {
        this.flag = flag;
    }

    public Activator getActivator() {
        return activator;
    }

    public void setActivator(Activator activator) {
        this.activator = activator;
    }

    public Mat getPart3() {
        return part3;
    }

    public void setPart3(Mat part3) {
        this.part3 = part3;
    }

    public Mat getGrad() {
        return grad;
    }

    public void setGrad(Mat grad) {
        this.grad = grad;
    }

    public int getWeightSize() {
        return weightSize;
    }

    public void setWeightSize(int weightSize) {
        this.weightSize = weightSize;
    }

    public Mat getWeight() {
        return weight;
    }

    public void setWeight(Mat weight) {
        this.weight = weight;
    }

    public Mat getInput() {
        return input;
    }

    public void setInput(Mat input) {
        this.input = input;
    }

    public Mat getActivatedOutput() {
        return activatedOutput;
    }

    public void setActivatedOutput(Mat activatedOutput) {
        this.activatedOutput = activatedOutput;
    }

    public Mat getOut() {
        return out;
    }

    public void setOut(Mat out) {
        this.out = out;
    }

    public Mat getPart1() {
        return part1;
    }

    public void setPart1(Mat part1) {
        this.part1 = part1;
    }

    public Mat getPart2() {
        return part2;
    }

    public void setPart2(Mat part2) {
        this.part2 = part2;
    }
}
