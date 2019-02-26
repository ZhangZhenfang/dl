package peer.afang.dl.test;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.util.Activator;
import peer.afang.dl.util.MatUtils;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/2/8 11:03
 */
public class ConvLayer {

    private Mat weight;
    private Mat input;
    private Mat out;
    private Mat activatedOut;
    private double bias;
    private ConvLayer previousLayer;
    private ConvLayer nextLayer;
    private Activator activator;
    private FCLayer fcLayer;
    private Mat delta;
    private Mat grad;
    public ConvLayer(int weightSize, Activator activator, int inputSize) {
        this.activator = activator;
        this.weight = new Mat(weightSize, weightSize, CvType.CV_32F);
        int outSize = (inputSize + 2 * 0 - weightSize) / 1 + 1;
        this.out = new Mat(outSize, outSize, CvType.CV_32F);
        this.activatedOut = new Mat(outSize, outSize, CvType.CV_32F);
        this.bias = 0;
        double[] data = new double[weightSize * weightSize];
        for (int i = 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        this.weight.put(0, 0, data);
    }

    public void computeOut() {
        MatUtils.conv(input, weight, out, 1, 0, 0);
        activator.activate(out, activatedOut);
    }

    public void computeGrad() {
        if (nextLayer == null) {
            this.delta = fcLayer.getInputDelta().reshape(1, 2);
            grad = MatUtils.conv(input, delta, 1, 1, 0);
        } else {
            Mat mat = MatUtils.paddingZeor(nextLayer.delta, 2, 2, 2, 2);
            Mat dst = new Mat();
            Core.flip(nextLayer.weight, dst, 1);
            Core.flip(dst, dst, 1);
            delta = MatUtils.conv(mat, dst, 1, 1, 0);
            grad = MatUtils.conv(input, delta, 1, 1, 0);
        }
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

    public Mat getOut() {
        return out;
    }

    public void setOut(Mat out) {
        this.out = out;
    }

    public Mat getActivatedOut() {
        return activatedOut;
    }

    public void setActivatedOut(Mat activatedOut) {
        this.activatedOut = activatedOut;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public ConvLayer getPreviousLayer() {
        return previousLayer;
    }

    public void setPreviousLayer(ConvLayer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public ConvLayer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(ConvLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public Activator getActivator() {
        return activator;
    }

    public void setActivator(Activator activator) {
        this.activator = activator;
    }

    public FCLayer getFcLayer() {
        return fcLayer;
    }

    public void setFcLayer(FCLayer fcLayer) {
        this.fcLayer = fcLayer;
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
}
