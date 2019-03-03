package peer.afang.dl.util;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;

/**
 * @author ZhangZhenfang
 * @date 2019/3/3 14:34
 */
public class Utils {
    public static int maxIndex(Mat mat) {
        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(mat);
        Point maxLoc = minMaxLocResult.maxLoc;
        return (int) maxLoc.x;
    }
}
