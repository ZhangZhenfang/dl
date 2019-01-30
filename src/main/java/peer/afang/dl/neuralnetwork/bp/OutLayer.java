package peer.afang.dl.neuralnetwork.bp;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * @author ZhangZhenfang
 * @date 2019/1/29 14:46
 */
public class OutLayer extends Layer {

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