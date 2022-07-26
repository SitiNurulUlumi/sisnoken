<!-- jQuery 3 -->
<script src="<?= base_url() ?>/assets/bower_components/jquery/dist/jquery.min.js"></script>
<!-- Bootstrap 3.3.7 -->
<script src="<?= base_url() ?>/assets/bower_components/bootstrap/dist/js/bootstrap.min.js"></script>
<!-- iCheck -->
<script src="<?= base_url() ?>/assets/plugins/iCheck/icheck.min.js"></script>
<!-- Sweetalert2 -->
<script src="<?= base_url() ?>/assets/bower_components/sweetalert2/sweetalert2.js"></script>

<script>
	$(document).ajaxStart(function() {
		Pace.restart();
	});

	<?php if ($this->session->flashdata()) : ?>
		var delay = alertify.get('notifier', 'delay');
		alertify.set('notifier', 'delay', 4);
		alertify.set('notifier', 'position', 'top-right');

		<?php if ($this->session->flashdata('msg_error')) : ?>
			alertify.error('<?= $this->session->flashdata('msg_error'); ?>');
		<?php endif; ?>
		<?php if ($this->session->flashdata('msg_success')) : ?>
			alertify.success('<?= $this->session->flashdata('msg_error'); ?>');
		<?php endif; ?>
	<?php endif; ?>
</script>

<script>
	$(function() {
		$('input').iCheck({
			checkboxClass: 'icheckbox_square-blue',
			radioClass: 'iradio_square-blue',
			increaseArea: '20%' /* optional */
		});
	});
</script>

<script>
	$(window).bind("load", function() {
		window.setTimeout(function() {
			$(".alert").fadeTo(500, 0).slideUp(500, function() {
				$(this).remove();
			});
		}, 4000);
	});
</script>

</body>

</html>