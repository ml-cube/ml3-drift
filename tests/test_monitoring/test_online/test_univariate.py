import numpy as np
from ml3_drift.monitoring.algorithms.online.adwin import ADWIN
from ml3_drift.monitoring.algorithms.online.kswin import KSWIN
from ml3_drift.monitoring.algorithms.online.page_hinkley import PageHinkley


class TestContinuousUnivariateOnlineAlgorithms:
    def test_kswin_univariate_one_drift(self, abrupt_univariate_online_drift_info):
        data_stream, drift_point_1 = abrupt_univariate_online_drift_info
        kswin_detector = KSWIN(p_value=0.0001)
        kswin_detector.fit(data_stream)
        output = kswin_detector.detect(data_stream)
        
        # Get output elements where drift_detected is True
        drift_detected_elements = [elem for elem in output if elem.drift_detected]
        
        # Get indices where drift was detected
        drift_detected_indices = [i for i, elem in enumerate(output) if elem.drift_detected]
        
        assert any([elem.drift_detected for elem in output])
        # Verify we have detected drift elements
        assert len(drift_detected_elements) > 0
        
        # Verify that the first detected drift occurs at or after the drift point
        first_drift_index = min(drift_detected_indices)
        assert first_drift_index >= drift_point_1, f"First drift detected at index {first_drift_index}, but expected at or after {drift_point_1}"
    
    def test_kswin_univariate_two_drift(self, abrupt_univariate_online_bidrift_info):
        data_stream, drift_point_1, drift_point_2 = abrupt_univariate_online_bidrift_info
        data_stream = np.array(data_stream).reshape(-1, 1)
        kswin_detector = KSWIN(p_value=0.0001)
        kswin_detector.fit(data_stream)
        output = kswin_detector.detect(data_stream)
        
        # Get output elements where drift_detected is True
        drift_detected_elements = [elem for elem in output if elem.drift_detected]
        
        # Get indices where drift was detected
        drift_detected_indices = [i for i, elem in enumerate(output) if elem.drift_detected]
        
        assert any([elem.drift_detected for elem in output])
        # Verify we have detected drift elements
        assert len(drift_detected_elements) == 2

        # Verify that the first detected drift occurs at or after the drift point
        first_drift_index = min(drift_detected_indices)
        second_drift_index = max(drift_detected_indices)
        assert first_drift_index >= drift_point_1 and first_drift_index < drift_point_2, f"First drift detected at index {first_drift_index}, but expected at or after {drift_point_1} and before {drift_point_2}"
        assert second_drift_index >= drift_point_2, f"Second drift detected at index {second_drift_index}, but expected at or after {drift_point_2}"
        

    def test_adwin_univariate_one_drift(self, abrupt_univariate_online_drift_info):
        data_stream, drift_point_1 = abrupt_univariate_online_drift_info

        adwin_detector = ADWIN()
        adwin_detector.fit(data_stream)
        output = adwin_detector.detect(data_stream)

        # Get output elements where drift_detected is True
        drift_detected_elements = [elem for elem in output if elem.drift_detected]
        
        # Get indices where drift was detected
        drift_detected_indices = [i for i, elem in enumerate(output) if elem.drift_detected]
        
        assert any([elem.drift_detected for elem in output])
        # Verify we have detected drift elements
        assert len(drift_detected_elements) > 0
        
        # Verify that the first detected drift occurs at or after the drift point
        first_drift_index = min(drift_detected_indices)
        assert first_drift_index >= drift_point_1, f"First drift detected at index {first_drift_index}, but expected at or after {drift_point_1}"
    
    def test_adwin_univariate_two_drift(self, abrupt_univariate_online_bidrift_info):
        data_stream, drift_point_1, drift_point_2 = abrupt_univariate_online_bidrift_info
        data_stream = np.array(data_stream).reshape(-1, 1)
        adwin_detector = ADWIN()
        adwin_detector.fit(data_stream)
        output = adwin_detector.detect(data_stream)

        # Get output elements where drift_detected is True
        drift_detected_elements = [elem for elem in output if elem.drift_detected]
        
        # Get indices where drift was detected
        drift_detected_indices = [i for i, elem in enumerate(output) if elem.drift_detected]
        
        assert any([elem.drift_detected for elem in output])
        # Verify we have detected drift elements
        assert len(drift_detected_elements) == 2

        # Verify that the first detected drift occurs at or after the drift point
        first_drift_index = min(drift_detected_indices)
        second_drift_index = max(drift_detected_indices)
        assert first_drift_index >= drift_point_1 and first_drift_index < drift_point_2, f"First drift detected at index {first_drift_index}, but expected at or after {drift_point_1} and before {drift_point_2}"
        assert second_drift_index >= drift_point_2, f"Second drift detected at index {second_drift_index}, but expected at or after {drift_point_2}"


    def test_page_hinkley_univariate_one_drift(self, abrupt_univariate_online_drift_info):
        data_stream, drift_point_1 = abrupt_univariate_online_drift_info
        hinkley_detector = PageHinkley()
        hinkley_detector.fit(data_stream)
        output = hinkley_detector.detect(data_stream)

        # Get output elements where drift_detected is True
        drift_detected_elements = [elem for elem in output if elem.drift_detected]
        
        # Get indices where drift was detected
        drift_detected_indices = [i for i, elem in enumerate(output) if elem.drift_detected]
        
        assert any([elem.drift_detected for elem in output])
        # Verify we have detected drift elements
        assert len(drift_detected_elements) > 0
        
        # Verify that the first detected drift occurs at or after the drift point
        first_drift_index = min(drift_detected_indices)
        assert first_drift_index >= drift_point_1, f"First drift detected at index {first_drift_index}, but expected at or after {drift_point_1}"

    def test_page_hinkley_univariate_two_drift(self, abrupt_univariate_online_bidrift_info):
        data_stream, drift_point_1, drift_point_2 = abrupt_univariate_online_bidrift_info
        data_stream = np.array(data_stream).reshape(-1, 1)
        hinkley_detector = PageHinkley()
        hinkley_detector.fit(data_stream)
        output = hinkley_detector.detect(data_stream)

        # Get output elements where drift_detected is True
        drift_detected_elements = [elem for elem in output if elem.drift_detected]
        
        # Get indices where drift was detected
        drift_detected_indices = [i for i, elem in enumerate(output) if elem.drift_detected]
        
        assert any([elem.drift_detected for elem in output])
        # Verify we have detected drift elements
        assert len(drift_detected_elements) == 2

        # Verify that the first detected drift occurs at or after the drift point
        first_drift_index = min(drift_detected_indices)
        second_drift_index = max(drift_detected_indices)
        assert first_drift_index >= drift_point_1 and first_drift_index < drift_point_2, f"First drift detected at index {first_drift_index}, but expected at or after {drift_point_1} and before {drift_point_2}"
        assert second_drift_index >= drift_point_2, f"Second drift detected at index {second_drift_index}, but expected at or after {drift_point_2}"
