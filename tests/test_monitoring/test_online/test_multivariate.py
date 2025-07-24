from ml3_drift.monitoring.algorithms.online.adwin import ADWIN
from ml3_drift.monitoring.algorithms.online.kswin import KSWIN
from ml3_drift.monitoring.algorithms.online.page_hinkley import PageHinkley
class TestContinuousMultivariateOnlineAlgorithms:
    def test_kswin_multivariate(self, abrupt_multivariate_drift_info):
        streaming_data, drift_point_1, _ = abrupt_multivariate_drift_info
        
        kswin_detector = KSWIN()
        kswin_detector.fit(streaming_data[:drift_point_1])
        output = kswin_detector.detect(streaming_data[drift_point_1:])
        assert any([elem.drift_detected for elem in output])
        # asser drift_point_1 is the first drifted sample
        

        assert kswin_detector.drift_detected is False
    def test_adwin_multivariate(self, abrupt_multivariate_drift_info):
        streaming_data, drift_point_1, _ = abrupt_multivariate_drift_info
        
        adwin_detector = ADWIN()
        adwin_detector.fit(streaming_data[:drift_point_1])
        output = adwin_detector.detect(streaming_data[drift_point_1:])
        assert any([elem.drift_detected for elem in output])
        # asser drift_point_1 is the first drifted sample
        

        assert adwin_detector.drift_detected is False
    def test_page_hinkley_multivariate(self, abrupt_multivariate_drift_info):
        streaming_data, drift_point_1, _ = abrupt_multivariate_drift_info
        
        ph_detector = PageHinkley()
        ph_detector.fit(streaming_data[:drift_point_1])
        output = ph_detector.detect(streaming_data[drift_point_1:])
        assert any([elem.drift_detected for elem in output])
        # asser drift_point_1 is the first drifted sample
        

        assert ph_detector.drift_detected is False


        

