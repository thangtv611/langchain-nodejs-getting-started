function executor(fn) {
    (async () => {
        try {
            fn();
        } catch (err) {
            console.error('[ERROR]: ', err);
            process.exit(0);
        }
    })();
}

module.exports = {
    executor,
};
