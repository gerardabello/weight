var React = require('react');

import LayerStats from './layer-stats.jsx';

const styles = {
  root:{
    border: '1px solid black',
  },
  image:{
    height: '60px',
    margin: '2px',
    imageRendering: 'pixelated',
    maxWidth: '100%',
  },
};

var Layer = React.createClass({
  render(){
    let layer = this.props.layer;

    let weight = [];

    if (layer.WeightsStats != null){
        weight = (
          <div>
            <h3>Weights</h3>
            <LayerStats stats={layer.WeightsStats}/>
            {layer.WeightsImg.map((b64img, i) =>
              <img key={i} style={styles.image} src={"data:image/png;base64," + b64img}></img>
            )}


            <h3>Bias</h3>
            <LayerStats stats={layer.BiasStats}/>
            {layer.BiasImg.map((b64img, i) =>
              <img key={i} style={styles.image} src={"data:image/png;base64," + b64img}></img>
            )}
          </div>
        )
    }

    return(
        <div style={styles.root}>
            <h2>{layer.ID}</h2>
            <h3>Out</h3>
            <LayerStats stats={layer.OutStats}/>
            {layer.OutImg.map((b64img, i) =>
              <img key={i} style={styles.image} src={"data:image/png;base64," + b64img}></img>
            )}

            <h3>Grad</h3>
            <LayerStats stats={layer.ErrStats}/>
            {layer.ErrImg.map((b64img, i) =>
              <img key={i} style={styles.image} src={"data:image/png;base64," + b64img}></img>
            )}

            {weight}

          </div>

    );

  }
});

export default Layer
