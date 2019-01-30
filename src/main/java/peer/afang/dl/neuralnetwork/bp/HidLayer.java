package peer.afang.dl.neuralnetwork.bp;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * @author ZhangZhenfang
 * @date 2019/1/29 14:45
 */
public class HidLayer extends Layer {

    public HidLayer(Mat input, int outputLength) {
        super(input, outputLength);
    }
    @Override
    void computeOut() {
//        System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
//        System.out.println(getInput().dump());
//        System.out.println(getWeight().dump());
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

