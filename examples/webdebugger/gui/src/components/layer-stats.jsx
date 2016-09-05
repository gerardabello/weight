var React = require('react');

const styles = {
};

var LayerStats = React.createClass({
  render(){
    return(
        <div>
          <span> Mean: {this.props.stats.Mean.toFixed(4)}</span>
          <span> StdDev: {this.props.stats.StDev.toFixed(4)}</span>
          <span> Min: {this.props.stats.Min.toFixed(4)}</span>
          <span> Max: {this.props.stats.Max.toFixed(4)}</span>
        </div>
    );

  }
});

export default LayerStats
