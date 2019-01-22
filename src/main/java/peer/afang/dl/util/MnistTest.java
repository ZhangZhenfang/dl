package peer.afang.dl.util;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.neuralnetwork.Bp;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * @author ZhangZhenfang
 * @date 19/1/22 9:46
 */
public class MnistTest {

    public static void main(String[] args) {
        MnistReader trainMnistReader = new MnistReader(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
        Bp bp = new Bp(784, 2, new int[]{300, 300}, 10);
        int times = 0;
        while(times < 20) {
            long startTime = System.currentTimeMillis();
            while(true) {
                double[] nextImage = trainMnistReader.getNextImage(true);
                if (nextImage == null) {
                    break;
                }
                Mat input = new Mat(1, 784, CvType.CV_32F);
                input.put(0, 0, nextImage);
                bp.newForward(input);

                Mat label = getNextLabel(trainMnistReader);
                bp.newBackPropagation(input, label, 0.2);
            }
            trainMnistReader.open(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
            times++;
            System.out.print("epoch " + times + " : ");
            System.out.println("cost " + (System.currentTimeMillis() - startTime) / 1000 + "s");
            test(bp);
        }
    }

    /**
     * 测试
     * @param bp
     */
    public static void test(Bp bp) {
        MnistReader testMnistReader = new MnistReader(MnistReader.TEST_IMAGES_FILE, MnistReader.TEST_LABELS_FILE);
        int numberOfTest = 0;
        int right = 0;
        int error = 0;
        while(true) {
            double[] nextImage = testMnistReader.getNextImage(true);
            if (nextImage == null) {
                break;
            }
            Mat input = new Mat(1, 784, CvType.CV_32F);
            input.put(0, 0, nextImage);
            bp.newForward(input);

            Mat label = getNextLabel(testMnistReader);
            int[] predict = indexOfMax(bp.getOutputLayer());
            int[] real = indexOfMax(label);
            numberOfTest++;
            if (predict[0] == real[0] && predict[1] == real[1]) {
                right++;
            } else {
                error++;
            }
        }
        System.out.print(numberOfTest + " " + right + " " + error + "  ");
        System.out.println("correct ratio : " + (double) right / numberOfTest);
    }

    public static Mat getNextLabel(MnistReader reader) {
        Mat label = new Mat(10, 1, CvType.CV_32F);
        double[] nextLabel = reader.getNextLabel();
        label.put(0, 0, nextLabel);
        return label;
    }

    /**
     * 获取矩阵m中最大值的坐标
     * @param m
     * @return
     */
    public static int[] indexOfMax(Mat m) {
        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(m);
        int[] result = new int[2];
        result[0] = (int) minMaxLocResult.maxLoc.x;
        result[1] = (int) minMaxLocResult.maxLoc.y;
        return result;
    }
    /**
     * draw a gray picture and the image format is JPEG.
     *
     * @param pixelValues pixelValues and ordered by column.
     * @param width       width
     * @param high        high
     * @param fileName    image saved file.
     */
    public static void drawGrayPicture(int[] pixelValues, int width, int high, String fileName) throws IOException {
        BufferedImage bufferedImage = new BufferedImage(width, high, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < high; j++) {
                int pixel = 255 - pixelValues[i * high + j];
                // r = g = b 时，正好为灰度
                int value = pixel + (pixel << 8) + (pixel << 16);
                bufferedImage.setRGB(j, i, value);
            }
        }
        ImageIO.write(bufferedImage, "JPEG", new File(fileName));
    }
}
