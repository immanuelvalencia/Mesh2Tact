"""Retrieve candidate publisher-deposited Crossref metadata for manual review."""
import concurrent.futures, json, urllib.request, urllib.parse, time
from pathlib import Path

TITLES = [
'Generation of GelSight Tactile Images for Sim2Real Learning',
'Taxim: An Example-based Simulation Model for GelSight Tactile Sensors',
'TACTO: A Fast, Flexible, and Open-Source Simulator for High-Resolution Vision-Based Tactile Sensors',
'FOTS: A Fast Optical Tactile Simulator for Sim2Real Learning of Tactile-motor Robot Manipulation Skills',
'Tacchi: A Pluggable and Low Computational Cost Elastomer Deformation Simulator for Optical Tactile Sensors',
'TacSL: A Library for Visuotactile Sensor Simulation and Learning',
'TacFlex: Multimode Tactile Imprints Simulation for Visuotactile Sensors With Coating Patterns',
'Simulation of Vision-based Tactile Sensors with Efficiency-tunable Rendering',
'Vision-based tactile sensor design using physically based rendering',
'Tactile Gym 2.0: Sim-to-real Deep Reinforcement Learning for Comparing Low-cost High-Resolution Robot Touch',
'Grasp Stability Prediction with Sim-to-Real Transfer from Tactile Sensing',
'Sim-to-Real Transfer for Optical Tactile Sensing',
'Learning the sense of touch in simulation: a sim-to-real strategy for vision-based tactile sensing',
'GelSight: High-Resolution Robot Tactile Sensors for Estimating Geometry and Force',
'DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor With Application to In-Hand Manipulation',
'GelSlim: A High-Resolution, Compact, Robust, and Calibrated Tactile-sensing Finger',
'GelSlim 3.0: High-Resolution Measurement of Shape, Force and Slip in a Compact Tactile-Sensing Finger',
'Improved GelSight Tactile Sensor for Measuring Geometry and Slip',
'GelTip: A Finger-shaped Optical Tactile Sensor for Robotic Manipulation',
'The TacTip Family: Soft Optical Tactile Sensors with 3D-Printed Biomimetic Morphologies',
'OmniTact: A Multi-Directional High-Resolution Touch Sensor',
'AllSight: A Low-Cost and High-Resolution Round Tactile Sensor With Zero-Shot Learning Capability',
'DTact: A Vision-Based Tactile Sensor that Measures High-Resolution 3D Geometry Directly from Darkness',
'DigiTac: A DIGIT-TacTip Hybrid Tactile Sensor for Comparing Low-Cost High-Resolution Robot Touch',
'Tactile data generation and applications based on visuo-tactile sensors: A review',
'Robotic tactile perception of object properties: A review',
'Look, Feel, and Learn: Self-Supervised Multimodal Learning for Robotic Manipulation',
'More Than a Feeling: Learning to Grasp and Regrasp Using Vision and Touch',
'Touch and Go: Learning from Human-Collected Vision and Touch',
'The Feeling of Success: Does Touch Sensing Help Predict Grasp Outcomes?',
'Connecting Touch and Vision via Cross-Modal Prediction',
'Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World',
'Sim-to-Real Transfer of Robotic Control with Dynamics Randomization',
'Image Quality Assessment: From Error Visibility to Structural Similarity',
'The Unreasonable Effectiveness of Deep Features as a Perceptual Metric',
'Illumination for Computer Generated Pictures',
'A Reflectance Map Technique for Determining Surface Orientation from Image Intensity',
'SimTac: A Physics-Based Simulator for Vision-Based Tactile Sensing with Biomorphic Structures',
]
OUT=Path(__file__).resolve().parents[1]/'references'
OUT.mkdir(exist_ok=True)
def fetch(pair):
    index,title=pair
    url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query.title':title,'rows':2})
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mesh2TactBibliography/1.0'})
            with urllib.request.urlopen(req,timeout=45) as response: data=json.load(response)
            items=data['message']['items']
            return {'index':index,'query':title,'candidates':items}
        except Exception as e:
            if attempt==2:return {'index':index,'query':title,'error':str(e)}
            time.sleep(2)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,enumerate(TITLES,1)))
    (OUT/'crossref_candidates.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for result in results:
        print(result['index'], result['query'])
        for candidate in result.get('candidates',[]):
            print(' ', candidate.get('title'),candidate.get('DOI'),candidate.get('container-title'),candidate.get('published',candidate.get('issued')))
        if 'error' in result: print(result['error'])
