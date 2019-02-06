package peer.afang.dl.neuralnetwork;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.util.Activator;
import peer.afang.dl.util.MatUtils;
import peer.afang.dl.util.Relu;
import peer.afang.dl.util.Tanh;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/2/3 17:21
 */
public class NewConvLayer {

    private int weightSize;

    private Mat weight;

    private Mat input;

    private Mat activatedOutput;

    private Mat out;

    private Activator activator;
    private Mat part1;
    private Mat part2;
    private Mat part3;

    public NewConvLayer(int weightSize, int inputSize, double[] data, Activator activator) {
        this.activator = activator;
        this.weight = new Mat(weightSize, weightSize, CvType.CV_32F);
//        double[] data = new double[weightSize * weightSize];
//        for (int i = 0; i < data.length; i++) {
//            new Random().nextDouble();
//        }
        weight.put(0, 0, data);
        int outSize = (inputSize + 2 * 0 - weightSize) / 1 + 1;
        this.out = new Mat(outSize, outSize, CvType.CV_32F);
        this.activatedOutput = new Mat(outSize, outSize, CvType.CV_32F);
    }

    public void computeOut() {
        Mat t = new Mat();
        Core.rotate(weight, t, 1);
//        (input.rows() + 2 * 0 - weight.rows()) / 1 + 1;
        MatUtils.conv(input, t, out, 1, 0, 0);
        System.out.println("************************************");
        System.out.println(input.dump());
        System.out.println(t.dump());
        System.out.println(out.dump());
        activator.activate(out, activatedOutput);
        System.out.println(activatedOutput.dump());
    }

    public void computeGrad(Mat lastPart1, Mat lastPart2, Mat lastWeight) {
        Mat ttt;
        Mat grad1partInpadweight = MatUtils.paddingZeor(lastWeight, 1, 1, 1, 1);
        Mat t5 = new Mat();
        Core.multiply(lastPart1, lastPart2, t5);
        Mat grad1partIn = new Mat();
        Core.rotate(t5, grad1partIn, 1);

        ttt = new Mat();
        Core.rotate(grad1partIn, ttt, 1);
        part1 = MatUtils.conv(grad1partInpadweight, ttt, 1, 0, 0);
        part2 = new Tanh().derivative(out);
        part3 = input;
        Mat t6 = new Mat();
        Core.multiply(part1, part2, t6);
        System.out.println(t6.dump());
        Mat t7 = new Mat();
        Core.rotate(t6, t7, 1);
        ttt = new Mat();
        Core.rotate(t7, ttt, 1);
        Mat t8 = MatUtils.conv(part3, ttt, 1, 0, 0);
        Mat grad1 = new Mat();
        Core.rotate(t8, grad1, 1);
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
